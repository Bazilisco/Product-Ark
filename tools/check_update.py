# tools/check_update.py
import subprocess, sys

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# tenta comparar HEAD com origin/main
run(["git", "fetch", "origin", "main"])
out = run(["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"]).stdout.strip()

try:
    left, right = map(int, out.split())
except Exception:
    print("Não foi possível verificar atualizações (sem remote 'origin' ou branch 'main').")
    sys.exit(0)

if right > 0:
    print(f"Atenção: existem {right} commit(s) novos no GitHub. Execute:")
    print("  git pull --ff-only")
else:
    print("Repositório já está atualizado.")
