# חיבור ל-GitHub דרך SSH

פרויקט זה מוגדר לשימוש ב-SSH (לא HTTPS) לגישה ל-GitHub.

## בדיקה: האם SSH פעיל?

```bash
# בדיקה שהמפתח קיים
ls -la ~/.ssh/id_ed25519*.pub

# בדיקת חיבור ל-GitHub
ssh -T git@github.com
```

תוצאה צפויה: `Hi USERNAME! You've successfully authenticated...`

---

## אם עדיין אין מפתח SSH

```bash
# יצירת מפתח (החלף את המייל)
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_github

# הצגת המפתח להעתקה ל-GitHub
cat ~/.ssh/id_ed25519_github.pub
```

**הוספה ל-GitHub:**
1. GitHub → Settings → SSH and GPG keys
2. New SSH key
3. הדבק את התוכן של `id_ed25519_github.pub`

---

## הגדרת Git לשימוש ב-SSH

הפרויקט מוגדר להפנות `https://github.com/` אוטומטית ל-`git@github.com:`.

אם צריך לעדכן את שם המשתמש ב-remote (החלף USERNAME בשם המשתמש שלך ב-GitHub):
```bash
git remote set-url origin git@github.com:ofirbts/ofir-ai-brain.git
```

---

## Push ראשון

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```
