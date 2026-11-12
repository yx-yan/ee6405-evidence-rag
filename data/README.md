# Medical corpus

`corpus.jsonl` contains 339 patient-information passages from the medical WikiDoc data prepared in
`MedQuAD_WikiDoc_preprocess.ipynb`. Each row includes a document ID, title, section, text, disease
label, and source URL. The disease labels cover asthma, diabetes, hypertension, and combinations of
those conditions.

`dataset.csv` contains the associated question/evaluation rows. The graph RAG demo builds disease
and shared-term links directly from the corpus and does not use evaluation answers to construct its
index.

The corpus and dataset paths used by the app are configured in `configs/default.yaml`.
