# Weekly Shopping List

The purpose of this project is to assist users in weekly meal preperation process. At the beginning of the week, users will create a daily plan for what they want to eat each day on MyFitnessPal. After running, the app will connect to a MyFitnessPal account and get the meals for each day of the next 7 days. It will calculate the total quantities of each food item and send the user an email with the total amounts of each food required to complete the week.

# Installation

Insure that myfitnesspal, datetime, re, and smtplib are installed. If not, run the following commands:

```bash
pip install myfitnesspal
pip install datetime
pip install re
pip install smtplib
```

# Usage

To run the program, run the following command:

```bash
python3 weekly_shopping_list.py
```

The program will prompt you to log into your MyFitnessPal account in your browser of choice, enter your email, and create a Google App Password to send the shopping list using the link: https://myaccount.google.com/apppasswords

Upon entering the email and App Password information, the program will send you an email with the shopping list.

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.