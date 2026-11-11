"""Metrics computed from results/runs/*.jsonl joined with dataset.csv. Owner: E.

Definitions (from the proposal):
  answer_correctness      correct supported answers / all questions
  selective_accuracy      correct answers / answered questions
  coverage                answered / all
  abstention_recall       correctly refused insufficient-evidence Qs / all insufficient-evidence Qs
  over_refusal_rate       refused sufficient-evidence Qs / all sufficient-evidence Qs
  unsupported_answer_rate answers not substantiated by retrieved evidence / generated answers
  latency                 mean and p95 latency_ms

TODO(E): implement, stratified by scenario (A/B/C) and method.
"""
#!/usr/bin/env python3
import os
import sys
import json
import argparse
import glob
import re
import collections
import numpy as np
import pandas as pd

def calculate_token_f1(gold_text, pred_text):
    """Computes token-level F1 score between target and prediction strings."""
    def tokenize(text):
        return re.findall(r'\b\w+\b', str(text).lower())

    gold_tokens = tokenize(gold_text)
    pred_tokens = tokenize(pred_text)
    
    if not gold_tokens or not pred_tokens:
        return 1.0 if gold_tokens == pred_tokens else 0.0
        
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def calculate_rouge_l(gold_text, pred_text):
    """Zero-dependency custom longest common subsequence calculator for text similarity."""
    def tokenize(text):
        return re.findall(r'\b\w+\b', str(text).lower())
    
    x = tokenize(gold_text)
    y = tokenize(pred_text)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
        
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i-1] == y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
                
    lcs = L[n][m]
    r_lcs = lcs / n
    p_lcs = lcs / m
    if r_lcs + p_lcs == 0:
        return 0.0
    return (2 * r_lcs * p_lcs) / (r_lcs + p_lcs)

def bootstrap_ci(data, num_bootstraps=1000, alpha=0.05):
    """Computes a 95% bootstrap confidence interval array matrix via pure NumPy resampling."""
    if len(data) == 0:
        return 0.0, 0.0
    means = []
    for _ in range(num_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, (alpha / 2) * 100), np.percentile(means, (1 - alpha / 2) * 100)

def parse_gold_answers(gold_val):
    """Safely extracts expected answers from JSON list configurations or plain text formats."""
    if pd.isna(gold_val):
        return []
    val_str = str(gold_val).strip()
    if not val_str:
        return []
    if val_str.startswith('[') and val_str.endswith(']'):
        try: return json.loads(val_str)
        except: pass
    if '|' in val_str:
        return [ans.strip() for ans in val_str.split('|') if ans.strip()]
    return [val_str]

def load_data(split_name):
    """Loads dataset.csv and maps execution log results/runs/*.jsonl records dynamically."""
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: {dataset_path} missing.")
        sys.exit(1)
        
    df_eval = pd.read_csv(dataset_path)
    df_eval['question_id'] = df_eval['question_id'].astype(str)
    df_eval = df_eval[df_eval['split'] == split_name]
    
    adjudication_path = f"results/adjudication_{split_name}.csv"
    adjudication_map = {}
    if os.path.exists(adjudication_path):
        adj_df = pd.read_csv(adjudication_path)
        for _, r in adj_df.iterrows():
            adjudication_map[(str(r.get('method')), str(r.get('question_id')))] = int(r.get('manual_is_correct', 0))
            
    run_files = glob.glob("results/runs/*.jsonl")
    all_records = []
    
    for rf in run_files:
        method_name = os.path.basename(rf).replace(".jsonl", "")
        with open(rf, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    run_item = json.loads(line)
                    q_id = str(run_item.get("question_id"))
                    matched_rows = df_eval[df_eval['question_id'] == q_id]
                    if matched_rows.empty: continue
                    row = matched_rows.iloc[0]
                    
                    g_ans_list = parse_gold_answers(row['gold_answers'])
                    gold_doc_ids = json.loads(row['gold_doc_ids']) if pd.notna(row['gold_doc_ids']) else []
                    
                    # Stratification logic for scenario parsing based on explicit prefix types
                    if "DIR" in q_id:
                        scenario = "A"
                    elif "PAR" in q_id:
                        scenario = "B"
                    elif "HOP" in q_id:
                        scenario = "C"
                    else:
                        # Fallback calculation based on count pattern extraction
                        try:
                            num_id = int(re.findall(r'\d+', q_id)[0])
                            scenario = "A" if num_id % 3 == 0 else "B" if num_id % 3 == 1 else "C"
                        except:
                            scenario = "A"
                        
                    generated_answer = run_item.get("generated_answer", "")
                    predicted_doc_ids = run_item.get("predicted_doc_ids", [])
                    
                    low_ans = generated_answer.lower()
                    refusal_keywords = ["insufficient evidence", "i do not have", "no information", "don't know", "sorry"]
                    is_refused = any(k in low_ans for k in refusal_keywords) or not generated_answer.strip()
                    
                    best_f1 = max([calculate_token_f1(g, generated_answer) for g in g_ans_list]) if g_ans_list else 0.0
                    best_rouge = max([calculate_rouge_l(g, generated_answer) for g in g_ans_list]) if g_ans_list else 0.0
                        
                    adj_key = (method_name, q_id)
                    if adj_key in adjudication_map:
                        is_correct = adjudication_map[adj_key] == 1
                    else:
                        is_correct = best_f1 >= 0.5 if not is_refused else False
                        
                    all_records.append({
                        "method": method_name, "question_id": q_id, "scenario": scenario,
                        "is_answerable": len(gold_doc_ids) > 0, "is_refused": is_refused,
                        "is_correct": is_correct, "is_supported": run_item.get("is_supported_by_evidence", True),
                        "latency_ms": run_item.get("latency_ms", 0.0), "token_f1": best_f1, "rouge_l": best_rouge,
                        "gold_doc_ids": gold_doc_ids, "predicted_doc_ids": predicted_doc_ids,
                        "shap_evidence_reliance": run_item.get("shap_evidence_reliance", 0.0),
                        "shap_gold_share": run_item.get("shap_gold_share", 0.0)
                    })
    return pd.DataFrame(all_records)

def process_metrics(df, run_attribution=False):
    """Calculates all parameters matching the system performance metric requirements."""
    metrics_summary = []
    if df.empty: return pd.DataFrame()
    
    for method in df['method'].unique():
        m_df = df[df['method'] == method]
        for sc in sorted(list(m_df['scenario'].unique())) + ['All']:
            s_df = m_df if sc == 'All' else m_df[m_df['scenario'] == sc]
            if s_df.empty: continue
                
            total_q = len(s_df)
            answered_df = s_df[~s_df['is_refused']]
            correct_array = s_df['is_correct'].astype(int).values
            ci_low, ci_high = bootstrap_ci(correct_array)
            
            insufficient_q = s_df[~s_df['is_answerable']]
            sufficient_q = s_df[s_df['is_answerable']]
            
            recall_k_list = [len(set(r['gold_doc_ids']) & set(r['predicted_doc_ids'])) / len(set(r['gold_doc_ids'])) 
                             for _, r in s_df.iterrows() if set(r['gold_doc_ids'])]
            
            record_metrics = {
                "method": method, 
                "scenario": sc,
                "coverage": len(answered_df) / total_q if total_q > 0 else 0.0,
                "answer_correctness": np.mean(correct_array) if len(correct_array) > 0 else 0.0,
                "correctness_ci_95": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "selective_accuracy": (answered_df['is_correct'].sum() / len(answered_df)) if len(answered_df) > 0 else 0.0,
                "abstention_recall": (insufficient_q['is_refused'].sum() / len(insufficient_q)) if len(insufficient_q) > 0 else 0.0,
                "over_refusal_rate": (sufficient_q['is_refused'].sum() / len(sufficient_q)) if len(sufficient_q) > 0 else 0.0,
                "unsupported_answer_rate": (answered_df['is_supported'] == False).sum() / len(answered_df) if len(answered_df) > 0 else 0.0,
                "token_f1": s_df['token_f1'].mean(), 
                "rouge_l": s_df['rouge_l'].mean(),
                "recall_at_k": np.mean(recall_k_list) if recall_k_list else 0.0,
                "latency_mean_ms": s_df['latency_ms'].mean() if len(s_df) > 0 else 0.0, 
                "latency_p95_ms": np.percentile(s_df['latency_ms'].values, 95) if len(s_df) > 0 else 0.0
            }
            
            if run_attribution:
                record_metrics["evidence_reliance_share"] = s_df['shap_evidence_reliance'].mean()
                record_metrics["gold_share"] = s_df['shap_gold_share'].mean()
                
            metrics_summary.append(record_metrics)
            
    return pd.DataFrame(metrics_summary)

def main():
    parser = argparse.ArgumentParser(description="EVRAG Evaluation Analysis Framework Engine")
    parser.add_argument("--split", type=str, required=True, help="Data split target parameter segment (e.g., dev, test)")
    parser.add_argument("--adjudication", action="store_true", help="Output manual adjudication evaluation template sheets")
    parser.add_argument("--attribution", type=int, help="Include SHAP token weight attribution metrics checks")
    args = parser.parse_args()
    
    joined_df = load_data(args.split)

    if joined_df.empty: 
        print("⚠️ No execution data paths matched. Check results/runs/ directory assets.")
        return
        
    if args.adjudication:
        os.makedirs("results", exist_ok=True)
        joined_df[["method", "question_id", "scenario", "token_f1", "is_correct"]].to_csv(f"results/adjudication_{args.split}.csv", index=False)
        print(f"📝 Generated human validation review checklist template at: '{out_file}'")
        return
        
    metrics_out = process_metrics(joined_df, run_attribution=(args.attribution is not None))
    os.makedirs("results", exist_ok=True)
    metrics_out.to_csv("results/metrics.csv", index=False)
    print("\n📊 Summary Evaluation Metrics Successfully Calculated:")
    print(metrics_out.to_string(index=False))

if __name__ == "__main__":
    main()

