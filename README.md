# machine-learning-model

Requirements:

```sh
pip install supabase pandas tensorflow scikit-learn matplotlib
```

Export env from `server`:

```sh
# eval $(sops -d ./../supabase/.env | grep -v '^#' | sed 's/^/export /')
sops -d ./../supabase/.env > .env
```
