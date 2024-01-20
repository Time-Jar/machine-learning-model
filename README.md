# machine-learning-model

Export env from `server`:

```sh
# eval $(sops -d ./../supabase/.env | grep -v '^#' | sed 's/^/export /')
sops -d ./../supabase/.env > .env
```

Build container using `docker build`.
