# הגדרת credentials לגישה ל-Google Drive

## צעד 1: צור פרויקט ב-Google Cloud

1. גלוש ל־[Google Cloud Console](https://console.cloud.google.com/)
2. לחץ על **Select a project** → **New Project**
3. הזן שם (למשל: `ofir-ai-brain`) ולחץ **Create**

## צעד 2: הפעל את Drive API

1. בתפריט השמאלי: **APIs & Services** → **Library**
2. חפש **Google Drive API**
3. לחץ **Enable**

## צעד 3: צור Service Account

1. **APIs & Services** → **Credentials**
2. **Create Credentials** → **Service account**
3. הזן שם (למשל: `ofir-brain-drive-reader`) → **Create and Continue**
4. אפשר לדלג על תפקידים (Role) → **Done**

## צעד 4: הורד מפתח JSON

1. ברשימת Service Accounts, לחץ על השורה שיצרת
2. לשונית **Keys** → **Add Key** → **Create new key**
3. בחר **JSON** → **Create**
4. קובץ JSON יורד למחשב שלך

## צעד 5: שיתוף התיקייה ב-Drive

1. העתק את ערך `client_email` מהקובץ (משהו כמו `xxx@xxx.iam.gserviceaccount.com`)
2. ב־Google Drive, עבור לתיקייה **ofir_brain**
3. קליק ימני → **Share**
4. הזן את כתובת ה־Service Account והוסף הרשאת **Viewer**

## צעד 6: שימוש בפרויקט

**אופציה א' – קובץ מקומי**
1. העתק את הקובץ שירד (או שנה ל־`credentials.json`) לתיקיית הפרויקט
2. הוסף ל־`.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
   ```

**אופציה ב' – JSON ב־env**
1. פתח את קובץ ה-JSON הורד
2. העתק את כל התוכן כשורה אחת (ללא רווחים/שורות)
3. הוסף ל־`.env`:
   ```
   GOOGLE_DRIVE_CREDENTIALS_JSON={"type":"service_account","project_id":"...",...}
   ```

## אבטחה

- **אל תשתף** את קובץ `credentials.json` או תעלה אותו ל-Git
- `credentials.json` כבר ב־`.gitignore`
- אל תעלה את המפתח ל-Drive או למקומות ציבוריים
