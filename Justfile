animate:
    python animate.py --db results-competition.db --umap-sample 100000
    
build-all-containers:
    just build-container ann-orthogonal/base .
    just build-container ann-orthogonal/team1 template
    just build-container ann-orthogonal/faiss competitors/faiss-hnsw

build-container name dir:
    docker build -t {{name}} {{dir}}

race-server:
    python -m http.server -d races
