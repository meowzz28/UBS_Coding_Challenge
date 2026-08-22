# UBS Adaptive API Gateway Challenge

A FastAPI solution for the Adaptive API Gateway challenge. The server accepts a
Base64-encoded V1 JSON model and returns the transformed V2 model.

## Transformation

`POST /solve` performs these changes:

- `adaptInput.user.id` becomes `adaptOutput.id`
- `adaptInput.user.fullName` becomes `adaptOutput.name`
- `adaptInput.action` is converted to lowercase
- priorities map as `LOW = 1`, `MEDIUM = 2`, and `HIGH = 3`

## Run locally

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

The API runs at <http://127.0.0.1:8000>. Interactive documentation is available
at <http://127.0.0.1:8000/docs>.

Test the challenge request:

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{"payload":"ewoJImFkYXB0SW5wdXQiOiB7CgkJInVzZXIiOiB7CgkJCSJpZCI6ICJVNDIiLAoJCQkiZnVsbE5hbWUiOiAiSmFuZSBEb2UiCgkJfSwKCQkiYWN0aW9uIjogIkNSRUFURSIsCgkJIm1ldGFkYXRhIjogewoJCQkicHJpb3JpdHkiOiAiSElHSCIKCQl9Cgl9Cn0="}'
```

Expected response:

```json
{
  "adaptOutput": {
    "id": "U42",
    "name": "Jane Doe",
    "action": "create",
    "priority": 3
  }
}
```

Run the tests with:

```bash
pytest
```

## Push to GitHub

From this project directory:

```bash
git init
git branch -M main
git add .
git commit -m "Build Adaptive API Gateway challenge server"
git remote add origin https://github.com/meowzz28/UBS_Coding_Challenge.git
git push -u origin main
```

GitHub will ask you to authenticate if your computer is not already signed in.
For HTTPS, use a personal access token instead of an account password, or sign in
with the GitHub CLI by running `gh auth login` before pushing.

If the GitHub repository already contains commits, clone it first or pull its
history before pushing; do not force-push over work you want to keep.

## Deploy on Render's free tier

The included `render.yaml` describes the service, start command, and health check.

1. Push this repository to GitHub.
2. Sign in at <https://dashboard.render.com>.
3. Choose **New > Blueprint**.
4. Connect GitHub and select `meowzz28/UBS_Coding_Challenge`.
5. Confirm the Blueprint and choose the **Free** instance if prompted.
6. Wait for the deploy to finish and copy the generated `onrender.com` URL.
7. Give the competition server `https://YOUR-SERVICE.onrender.com/solve`.

Every push to `main` will trigger a new deployment. The free service may sleep
after a period without traffic, so the first request after inactivity can be slow.

You can verify the deployed service with:

```bash
curl https://YOUR-SERVICE.onrender.com/health
```

## Adding more challenge questions

If the competition expects one callback URL, keeping the challenges in one repo
and adding new routes/modules is usually simpler than maintaining one repository
per question. If each question must expose its own `/solve` URL, create one folder
and one Render service per question, or use separate repositories as originally
planned.
