import myfitnesspal
import datetime
import re

def get_dates():
    ''' Get date, month, and year for the next 7 days 
    
    Returns:
        days (list(tuples)) - tuples of (year, month, day)
    '''

    # Get today's date
    today = datetime.date.today()

    # Get the next 7 days
    days = []
    for _ in range(7):
        days.append((today.year, today.month, today.day))
        today += datetime.timedelta(days=1)

    return days

def parse_meal_entries(entry):
    ''' Parse the meal entries from MFP

    Parameters:
        entry (myfitnesspal.models.Entry) - MFP entry object

    Returns:
        food_name (str) - name of the food
        quantity (float) - quantity of the food
        unit (str) - unit of measurement
    '''

    # Get the unit of measurement by searching past the last comma
    unit = re.search(r', (.+) (.+)$', entry.name).group(2)

    #Delete the quantity from the food name
    food_name = re.search(r'(.+),', entry.name).group(1)
                
    return food_name, float(entry.quantity), unit

def create_shopping_list():
    ''' Get data from MFP'''
    # Create a client
    client = myfitnesspal.Client()

    # Get the next 7 days
    days = get_dates()
    
    # Login and get data meal data for each day
    shopping_list = {}

    for day in days:
        for meals in client.get_date(*day).meals:
            for entry in meals.entries:
                # Parse the meal entries for each meal of each day
                food_name, quantity, unit = parse_meal_entries(entry)

                # Add the food to the shopping list. 
                # If the food is already in the list, add the quantity
                if food_name in shopping_list:
                    shopping_list[food_name][0] += quantity
                else:
                    shopping_list[food_name] = [quantity, unit]

    return shopping_list

def shopping_list_to_string(shopping_list):
    ''' Print the shopping list in a dataframe

    Parameters:
        shopping_list (dict) - dictionary of food items and quantities
    '''
    string = ''

    for food in shopping_list:
        string += f'{food} \n {shopping_list[food][0]} {shopping_list[food][1]} \n\n'

    return string