# Security Policy

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Supported scope

Security patches are only applied to the latest version on the repository's default branch. Older commits may not be updated.

## Reporting vulnerabilities

Do not post API keys, restart secrets, private tunnel URLs, Hugging Face tokens, or sensitive exploit details in public issues.

Prefer **GitHub → Security → Advisories → Report a vulnerability** to send a private report to the repository owner. If Private Vulnerability Reporting is not enabled, contact **Đăng Khoa <i.am@dangkhoa.dev>** privately.

A useful report should include:

- The affected version or commit.
- The relevant deployment configuration, with any secrets removed.
- Minimal reproduction steps.
- Expected impact and a suggested fix, if any.

## Handling leaked credentials

If a credential was ever committed, deleting the file in a new commit is not enough. Instead:

1. Revoke or rotate the credential immediately.
2. Remove the credential from the entire Git history when needed.
3. Check access logs and any related secrets.
4. Enable GitHub secret scanning and push protection.

Runtime files such as `.env`, `data/api_key.txt`, `data/restart_secret.txt`, logs, and caches are covered by `.gitignore`, but operators are still responsible for checking before every push.

## Security limitations

The project is designed for demos/experiments on Kaggle, not large-scale production infrastructure. Cloudflare Quick Tunnel creates a public URL; always keep API authentication enabled and never treat a random URL as a security mechanism.
