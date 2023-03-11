import mfp_functions as mfp
import smtplib
import datetime

if __name__ == "__main__":
    # Prompt the user to log into MyFitnessPal in their browser
    print('Log into MyFitnessPal in your browser and then press Enter.')
    input()

    # Prompt the user for their email address and password
    sender_email = input('Enter your email address: ')
    
    print('Enter your app password. If you don\'t have one, go to '
          'https://myaccount.google.com/apppasswords to create one.')
    
    sender_password = input('Enter your app password: ')

    # Create the shopping list
    print('Creating shopping list...')

    # get the current date and the date 7 days from now
    today = datetime.date.today()
    next_week = today + datetime.timedelta(days=7)
    subject = f'Shopping list for {today} to {next_week}'

    shopping_list = mfp.create_shopping_list()
    body = mfp.shopping_list_to_string(shopping_list)

    message = f'Subject: {subject}\n\n{body}'.encode('utf-8')

    print('Sending email...')

    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Log in to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)

    # Send the email
    server.sendmail(sender_email, sender_email, message)

    # Close the connection to the SMTP server
    server.quit()

    print('Done!')
