from flask import Flask, redirect, url_for, render_template, request, flash, session
import re
import os
import sqlite3
import hashlib
from base64 import decode
import json
from decimal import Decimal
from datetime import date, datetime,timedelta 
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder 
import random
from flask import Flask, render_template, request
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key='honsproject'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.globals.update(zip=zip)

# app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class_mapping = {
    0: {'label': 'aloo-gobi'},
    1: {'label': 'aloo-fry'},
    2: {'label': 'dum-aloo'},
    3: {'label': 'fish-curry'},
    4: {'label': 'ghevar'},
    5: {'label': 'green-chutney'},
    6: {'label': 'gulab-jamun'},
    7: {'label': 'idli'},
    8: {'label': 'jalebi'},
    9: {'label': 'chicken-seekh-kebab'},
    10: {'label': 'kheer'},
    11: {'label': 'kulfi'},
    12: {'label': 'bhature'}, 
    13: {'label': 'lassi'},
    14: {'label': 'mutton-curry'},
    15: {'label': 'onion-pakoda'},
    16: {'label': 'palak-paneer'},
    17: {'label': 'poha'},
    18: {'label': 'rajma-curry'},
    19: {'label': 'rasmalai'},
    20: {'label': 'samosa'},
    21: {'label': 'shahi-paneer'},
    22: {'label': 'white-rice'},
    23: {'label': 'bhindi-masala'},
    24: {'label': 'chicken-biryani'},
    25: {'label': 'chai'},
    26: {'label': 'chole'},
    27: {'label': 'coconut-chutney'},
    28: {'label': 'dal-tadka'},
    29: {'label': 'dosa'}
}

# Load nutrition datasets on startup
indian_food_df = None
fooddata_central_foods = None
fooddata_central_nutrients = None

def load_nutrition_datasets():
    """Load CSV datasets for nutrition lookup"""
    global indian_food_df, fooddata_central_foods, fooddata_central_nutrients
    
    try:
        # Load Indian Food Nutrition CSV
        indian_food_df = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
        print(f"✅ Loaded Indian Food Nutrition database: {len(indian_food_df)} dishes")
        
        # Load FoodData Central CSVs
        fooddata_central_foods = pd.read_csv('FoodData_Central_foundation_food_csv_2025-04-24/food.csv')
        fooddata_central_nutrients = pd.read_csv('FoodData_Central_foundation_food_csv_2025-04-24/food_nutrient.csv')
        print(f"✅ Loaded FoodData Central database: {len(fooddata_central_foods)} foods")
        
        return True
    except Exception as e:
        print(f"⚠️ Error loading nutrition datasets: {e}")
        return False

# Load datasets on app startup
load_nutrition_datasets()

def fetch_nutrition_from_csv(food_name):
    """
    Fetch nutrition data from CSV datasets
    Searches Indian Food Nutrition CSV first, then FoodData Central
    Returns dictionary with calories, protein, carbs, and fat, or None if not found
    """
    # Format food name for better matching
    formatted_name = food_name.replace('-', ' ').strip()
    
    # Try Indian Food Nutrition CSV first
    if indian_food_df is not None:
        # Search for food (case-insensitive, partial match)
        matches = indian_food_df[indian_food_df['Dish Name'].str.contains(formatted_name, case=False, na=False)]
        
        if len(matches) > 0:
            # Use first match
            food_data = matches.iloc[0]
            print(f"✅ Found '{food_name}' in Indian Food database as '{food_data['Dish Name']}'")
            
            return {
                'calories': round(float(food_data['Calories (kcal)']), 2),
                'protein': round(float(food_data['Protein (g)']), 2),
                'carbs': round(float(food_data['Carbohydrates (g)']), 2),
                'fat': round(float(food_data['Fats (g)']), 2),
                'source': 'Indian Food Database'
            }
    
    # Try FoodData Central as fallback
    if fooddata_central_foods is not None and fooddata_central_nutrients is not None:
        # Search in FoodData Central
        matches = fooddata_central_foods[fooddata_central_foods['description'].str.contains(formatted_name, case=False, na=False)]
        
        if len(matches) > 0:
            food_data = matches.iloc[0]
            fdc_id = food_data['fdc_id']
            
            # Get nutrients for this food
            food_nutrients = fooddata_central_nutrients[fooddata_central_nutrients['fdc_id'] == fdc_id]
            
            # Extract key nutrients (IDs: 1008=Energy, 1003=Protein, 1005=Carbs, 1004=Fat)
            nutrients = {}
            for nutrient_id, key in [(1008, 'calories'), (1003, 'protein'), (1005, 'carbs'), (1004, 'fat')]:
                nutrient_data = food_nutrients[food_nutrients['nutrient_id'] == nutrient_id]
                if len(nutrient_data) > 0:
                    nutrients[key] = round(float(nutrient_data.iloc[0]['amount']), 2)
                else:
                    nutrients[key] = 0.0
            
            if nutrients.get('calories', 0) > 0:  # Valid nutrition data found
                print(f"✅ Found '{food_name}' in FoodData Central as '{food_data['description']}'")
                nutrients['source'] = 'FoodData Central'
                return nutrients
    
    # Not found in either database
    print(f"❌ '{food_name}' not found in nutrition databases")
    return None

def detect_and_visualize(img, model_path, class_mapping, confidence_threshold=0.25):
    model = YOLO(model_path)

    results = model.predict(source=img, conf=confidence_threshold)
    detected_items = [0]*30
    float_detections = results[0].boxes.xyxy.tolist()
    detections = [[int(value) for value in detection] for detection in float_detections]
    confidences = results[0].boxes.conf.tolist()
    float_classes = results[0].boxes.cls.tolist()
    classes = [int(value) for value in float_classes]

    total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    resized_img = cv2.resize(img, (800, 400))

    scaling_factor_x = 800 / img.shape[1]
    scaling_factor_y = 400 / img.shape[0]

    for i in range(len(detections)):
        box = detections[i]
        resized_box = [
            int(box[0] * scaling_factor_x),
            int(box[1] * scaling_factor_y),
            int(box[2] * scaling_factor_x),
            int(box[3] * scaling_factor_y)
        ]
        class_index = classes[i]
        class_info = class_mapping.get(class_index, {'label': 'unknown'})
        conf = confidences[i]
        if conf > 0.4:
            detected_items[class_index] += 1

            class_label = class_info['label']

            # Display food name and confidence on image
            cv2.putText(resized_img, f'{class_label} {conf:.2f}', (resized_box[0], resized_box[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.rectangle(resized_img, (resized_box[0], resized_box[1]), (resized_box[2], resized_box[3]), (255, 0, 255), 2)
    
    # Convert the OpenCV image to bytes
    _, result_image = cv2.imencode('.jpg', resized_img)
    result_bytes = result_image.tobytes()

    items_with_nutrition = []
    for i in range(30):
        if(detected_items[i] != 0):
            food_label = class_mapping[i].get('label')
            count = detected_items[i]
            
            # Fetch nutrition data from CSV databases
            nutrition_data = fetch_nutrition_from_csv(food_label)
            
            if nutrition_data:
                # Use data from CSV databases
                item_calories = nutrition_data['calories'] * count
                item_protein = nutrition_data['protein'] * count
                item_carbs = nutrition_data['carbs'] * count
                item_fat = nutrition_data['fat'] * count
                data_available = True
            else:
                # Data not available in databases
                item_calories = 0
                item_protein = 0
                item_carbs = 0
                item_fat = 0
                data_available = False
            
            # Update totals (only if data available)
            if data_available:
                total_nutrition['calories'] += item_calories
                total_nutrition['protein'] += item_protein
                total_nutrition['carbs'] += item_carbs
                total_nutrition['fat'] += item_fat
            
            items_with_nutrition.append({
                'label': food_label,
                'count': count,
                'calories': round(item_calories, 2) if data_available else None,
                'protein': round(item_protein, 2) if data_available else None,
                'carbs': round(item_carbs, 2) if data_available else None,
                'fat': round(item_fat, 2) if data_available else None,
                'data_available': data_available
            })
    
    # Round total nutrition values
    total_nutrition = {k: round(v, 2) for k, v in total_nutrition.items()}
    
    return result_bytes, total_nutrition, items_with_nutrition

@app.route('/')
@app.route('/first')
def first():
    return render_template('home.html')

@app.route('/predict')
def index1():
    return render_template('index1.html')

@app.route('/about1')
def about1():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index1.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index1.html', error="")

    if file and allowed_file(file.filename):
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        result_bytes, total_nutrition, items_with_nutrition = detect_and_visualize(img, r"best.pt", class_mapping)
        
        return render_template('index1.html', 
                             filename=f'data:image/jpg;base64,{base64.b64encode(result_bytes).decode()}', 
                             total_nutrition=total_nutrition, 
                             items_with_nutrition=items_with_nutrition, 
                             name=file.filename)

def get_connection():
	conn = sqlite3.connect('diet_recommendation.db')
	conn.row_factory=sqlite3.Row # to be able to reference by column name
	return conn

def pwd_security(passwd):
	"""A strong password must be at least 8 characters long
	   and must contain a lower case letter, an upper case letter,
	   and at least 3 digits.
	   Returns True if passwd meets these criteria, otherwise returns False.
	   """
	# check password length
	# check password for uppercase, lowercase and numeric chars
	hasupper = False	
	haslower = False
	digitcount = 0
	digit= False
	strong = False
	length = True
	special = False
	for c in passwd:
		if (c.isupper()==True):
			hasupper= True
		elif (c.islower()==True):
			haslower=True
		elif (c.isdigit()==True):
			digitcount+=1
			digit = True
		elif re.findall('[^A-Za-z0-9]',c):
			special = True
	if hasupper == True and haslower == True and digit == True and special == True:
		strong = True
	if len(passwd) <8:
		length = False
	return strong,haslower,hasupper,digit,length, special

def pwd_encode(pwd):
	secure_pwd =hashlib.md5(pwd.encode()).hexdigest()
	return secure_pwd

@app.route("/update_profile", methods =['GET','POST'] )
def edit_weight():
	if request.method == 'GET':

		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_id=?",(session['uid'],))
				u_data = cur.fetchone()
				name = u_data[1]
				age = u_data[4]
				weight = u_data[6]
				email = u_data[5]
				password = session['u_pass']
				gender = u_data[3]
				ft = u_data[7]
				inch = u_data[8]
				vegan = u_data[12]
				allergy = u_data[13]

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()
		

		return render_template('edit_profile.html', name= name,
													age = age,
													weight = weight,
													email = email,
													password = password,
													gender = gender,
													ft = ft,
													inch = inch,
													vegan = vegan,
													allergy = allergy)
	else:
		# name = u_data[1]
		# age = u_data[4]
		# weight = u_data[6]
		# email = u_data[5]
		# password = session['u_pass']
		# gender = u_data[3]
		# ft = u_data[7]
		# inch = u_data[8]
		# vegan = u_data[12]
		# allergy = u_data[13]

		return redirect(url_for('profile'))

@app.route("/register", methods = ['GET','POST'])
def citizen_register():
	if request.method == 'GET':
		return render_template('register.html')
	else:
		name = request.form['name']
		
		email = request.form['email']
		password = request.form['password']
		
		return render_template('profilesetup.html',name=name,email=email,password=password)

@app.route("/login", methods = ['GET','POST'])
def login():
	if 'uid' in session:
		return redirect('/index')
	if request.method == 'GET':
		return render_template('login.html')
	else:
		
		session['uid'] = 0
		email = request.form['email']
		password = request.form['password']
		secure_pwd = pwd_encode(password)
		msg=''
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_email=?",(email,))
				u_info= cur.fetchall()
				if not u_info:
					flash(f'The email address ({email}) that you entered does not exist in our database.')
					return redirect(url_for('login'))
				else:
					for row in u_info:
						session['uid'] = row[0]
						u_pass = row[2] 
						u_name = row[1]
						u_date = row[-1]
					
					if secure_pwd == u_pass:
						days = []
						flash(f'Your have successfully logged in as {u_name}')
						session['u_logged'] = True
						session['u_info'] = []
						session['u_pass'] = password 

						track_date = datetime.today().strftime ('%Y-%m-%d')
						sdate = datetime.strptime(u_date, '%Y-%m-%d').date()
						edate = datetime.strptime(track_date, '%Y-%m-%d').date()
						delta = edate - sdate     

						for i in range(delta.days + 1):
							day = sdate + timedelta(days=i)
							days.append(str(day))
							journey = len(days)

						try:
							with get_connection() as conn:
								cur = conn.cursor()
								cur2 = conn.cursor()
								cur2.execute("update user set u_journey=? where u_id=?", (journey,session['uid'],))
								conn.commit()

								cur.execute("select * from user where u_id=?",(session['uid'],))
								u_info = cur.fetchone()
				
								for row in u_info:
									session['u_info'].append(row)

						except sqlite3.Error as e:
							return (f'{e}')
						finally:
							conn.close()

						return redirect(url_for('index'))
					else:
						session.pop('uid',None)
						flash('Sorry the credentails you are using are invalid')
						return redirect(url_for('login'))

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

@app.route("/setup", methods = ['GET','POST'])
def profilesetup():
	if request.method == 'GET':
		return render_template('profilesetup.html')

	else:
		name = request.form['name']
		email = request.form['email']
		passwd = request.form['password']
		password = pwd_encode(passwd)
		age = int(request.form['age'])
		gender = request.form['gender']
		vegan = request.form['vegan']
		allergy = request.form['allergy']
		weight_lb = int(request.form['weight'])
		feet = int(request.form['feet'])
		inches = int(request.form['inches'])
		activity_level = request.form['activity']
		height_bmi = int((feet * 12) + inches)
		bmr = 0
		body_status = ""
		BMI =  weight_lb / (height_bmi*height_bmi) * 703
		bodyfat = 0

		if gender == "male":
			bmr = int((4.536 * weight_lb) + (15.88 * height_bmi) - (5 * age) + 5)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 16.2)
		else:
			bmr = int((4.536 * weight_lb) + (15.88 * height_bmi) - (5 * age) - 161)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 5.4)

		calorie = 0

		if activity_level == "sedentary":
			calorie = int(bmr*1.2)

		elif activity_level == "lightly active":
			calorie = int(bmr * 1.375)

		elif activity_level == "moderately active":
			calorie = int(bmr * 1.55)

		elif activity_level == "very active":
			calorie = int(bmr * 1.725)

		elif activity_level == "extra active":
			calorie = int(bmr * 1.9)

		if BMI < 18.5:
			body_status = "underweight"

		elif BMI >= 18.5 and BMI <= 24.9 :
			body_status = "healthy weight"

		elif BMI >= 25 and BMI <= 29.9 :
			body_status = "overweight"

		elif BMI >= 30 :
			body_status = "obese"

		protein = int(((calorie-500) * 0.30)/4)
		carb = int(((calorie-500)* 0.40)/4)
		fat = int(((calorie-500) * 0.30)/9)
		fiber = int(calorie/1000*14)
		journey = 1
		breakfast = int((calorie-500) * 0.30)
		snack = int((calorie-500)* 0.10)
		lunch = int((calorie-500)* 0.35)
		dinner = int((calorie-500)* 0.25)

		try:
			with get_connection() as conn:
				db = conn.cursor()
				db.execute("insert into user (u_username,u_email,u_password,u_age,u_gender,u_vegan,u_allergy,u_weight,u_feet,u_inches,u_bmi,u_activitylevel,u_protein,u_carb,u_fat,u_fiber,u_calories,u_journey,u_bodyfat,u_status,u_startdate) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(name,email,password,age,gender,vegan,allergy,weight_lb,feet,inches,int(BMI),activity_level,protein,carb,fat,fiber,calorie,journey,bodyfat,body_status,datetime.today().strftime ('%Y-%m-%d'),))
				conn.commit()
				flash('Successfully Registered')

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

	
		return redirect(url_for('login'))

@app.route("/profile", methods = ['GET','POST'])
def profile():
	if request.method == 'GET':
		uid = session['uid']
		try:
			with get_connection() as conn:
				db = conn.cursor()
				db.execute("select * from user where u_id=?",(uid,))
				u_info = db.fetchone()

				

				
		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

		return render_template('profile.html',u_info = u_info)

	else:
		return render_template('profile.html')

@app.route("/recommendation", methods = ['GET','POST'])
def recommendation():
	if request.method == 'GET':
		return render_template('recommendation.html')

	else:
		dataset = pd.read_csv('dietdataset.csv')

		dataset = pd.DataFrame(data=dataset.iloc[:,0:10].values,columns = ['meal_name','carb','meat','vege','fruit', 'type','vegan','allergy','time'])
		le = LabelEncoder()
		dataset_encoded = dataset.iloc[:,0:10]
		for i in dataset_encoded:
			dataset_encoded[i] = le.fit_transform(dataset_encoded[i])
			
			model = pickle.load(open('model1','rb'))

		bf_vege_input = []
		bf_meat_input = []
		bf_carb_input = []
		bf_fruit_input = []

		bf_vege = random.choice(request.form.getlist('vege'))
		bf_meat = random.choice(request.form.getlist('meat'))
		bf_carb = random.choice(request.form.getlist('carb'))
		bf_fruit = random.choice(request.form.getlist('fruit'))

		bf_vege_input.append(bf_vege)
		bf_meat_input.append(bf_meat)
		bf_carb_input.append(bf_carb)
		bf_fruit_input.append(bf_fruit)

		lunch_vege_input = []
		lunch_meat_input = []
		lunch_carb_input = []
		lunch_fruit_input = []

		lunch_vege = random.choice(request.form.getlist('vege'))
		lunch_meat = random.choice(request.form.getlist('meat'))
		lunch_carb = random.choice(request.form.getlist('carb'))
		lunch_fruit = random.choice(request.form.getlist('fruit'))

		lunch_vege_input.append(lunch_vege)
		lunch_meat_input.append(lunch_meat)
		lunch_carb_input.append(lunch_carb)
		lunch_fruit_input.append(lunch_fruit)

		snack_vege_input = []
		snack_meat_input = []
		snack_carb_input = []
		snack_fruit_input = []

		snack_vege = random.choice(request.form.getlist('vege'))
		snack_meat = random.choice(request.form.getlist('meat'))
		snack_carb = random.choice(request.form.getlist('carb'))
		snack_fruit = random.choice(request.form.getlist('fruit'))

		snack_vege_input.append(snack_vege)
		snack_meat_input.append(snack_meat)
		snack_carb_input.append(snack_carb)
		snack_fruit_input.append(snack_fruit)

		dinner_vege_input = []
		dinner_meat_input = []
		dinner_carb_input = []
		dinner_fruit_input = []

		dinner_vege = random.choice(request.form.getlist('vege'))
		dinner_meat = random.choice(request.form.getlist('meat'))
		dinner_carb = random.choice(request.form.getlist('carb'))
		dinner_fruit = random.choice(request.form.getlist('fruit'))

		dinner_vege_input.append(dinner_vege)
		dinner_meat_input.append(dinner_meat)
		dinner_carb_input.append(dinner_carb)
		dinner_fruit_input.append(dinner_fruit)
		
		type_breakfast = request.form.getlist('breakfast_dishes')
		type_lunch =  request.form.getlist('lunch_dishes')
		type_dinner = request.form.getlist('dinner_dishes')
		type_snack = request.form.getlist('snack_dishes')
		print(type_breakfast)
		allergy_input = []
		vegan_input = []
		allergy_input.append(session['u_info'][13])
		vegan_input.append(session['u_info'][12])

		print(type_breakfast,allergy_input,vegan_input)
		time_breakfast = ['Breakfast']
		time_snack = ['Snack']
		time_lunch = ['Lunch']
		time_dinner = ['Dinner']

		def input_encode(entry, room):
			meal = dataset.values.tolist()
			meal_encode = dataset_encoded.values.tolist()
			lists = []
			encode = []
    
			for i in entry:
				found = False
				for j in meal:
					if i == j[room]:
						lists.append(j)
						found = True
						break
			if not found:
				print(f"Warning: '{i}' not found in dataset column {room}")
    
			for j in lists:
				encode.append(meal_encode[meal.index(j)][room])
        
			if not encode:  # No match found
				encode = [0]  # Set a default value or handle accordingly

			return encode
#         return meal_encode[meal.index(j)][room]

		def input_decode(entry,room): 
			meal = dataset.values.tolist()
			meal_encode = dataset_encoded.values.tolist()
			lists = []
			decode = []   
			for i in entry:
				for j in meal_encode:
					if i==j[room]:
						lists.append(j)
						break
                 
			for j in lists: 
				decode.append(meal[meal_encode.index(j)][room])
        
			return decode

		bf_carb_encode = input_encode(bf_carb_input,1)[0]
		bf_meat_encode = input_encode(bf_meat_input,2)[0]
		bf_vege_encode = input_encode(bf_vege_input,3)[0]
		bf_fruit_encode = input_encode(bf_fruit_input,4)[0]

		lunch_carb_encode = input_encode(lunch_carb_input,1)[0]
		lunch_meat_encode = input_encode(lunch_meat_input,2)[0]
		lunch_vege_encode = input_encode(lunch_vege_input,3)[0]
		lunch_fruit_encode = input_encode(lunch_fruit_input,4)[0]

		dinner_carb_encode = input_encode(dinner_carb_input,1)[0]
		dinner_meat_encode = input_encode(dinner_meat_input,2)[0]
		dinner_vege_encode = input_encode(dinner_vege_input,3)[0]
		dinner_fruit_encode = input_encode(dinner_fruit_input,4)[0]

		snack_carb_encode = input_encode(snack_carb_input,1)[0]
		snack_meat_encode = input_encode(snack_meat_input,2)[0]
		snack_vege_encode = input_encode(snack_vege_input,3)[0]
		snack_fruit_encode = input_encode(snack_fruit_input,4)[0]
				
		breakfast_encode = input_encode(type_breakfast,5)[0]
		lunch_encode = input_encode(type_lunch,5)[0]
		snack_encode = input_encode(type_snack,5)[0]
		dinner_encode = input_encode(type_dinner,5)[0]

		vegan_encode = input_encode(vegan_input,6)[0]
		allergy_encode = input_encode(allergy_input,7)[0]
		bf_time_encode = input_encode(time_breakfast,8)[0]
		lunch_time_encode = input_encode(time_lunch,8)[0]
		snack_time_encode = input_encode(time_snack,8)[0]
		dinner_time_encode = input_encode(time_dinner,8)[0]

		bf_input = [bf_carb_encode,bf_meat_encode,bf_vege_encode,bf_fruit_encode,breakfast_encode,vegan_encode,allergy_encode,bf_time_encode]
		
		bf_result = model.predict([bf_input])
		bf_prediction = input_decode(bf_result,0)	
			
		lunch_input = [lunch_carb_encode,lunch_meat_encode,lunch_vege_encode,lunch_fruit_encode,lunch_encode,vegan_encode,allergy_encode,lunch_time_encode]
		lunch_result = model.predict([lunch_input])
		lunch_prediction = input_decode(lunch_result,0)	
			
		snack_input = [snack_encode,snack_encode,snack_encode,snack_encode,snack_encode,vegan_encode,allergy_encode,snack_time_encode]
		snack_result = model.predict([snack_input])
		snack_prediction = input_decode(snack_result,0)	
						
		dinner_input = [dinner_encode,dinner_encode,dinner_encode,dinner_encode,dinner_encode,vegan_encode,allergy_encode,dinner_time_encode]
		dinner_result = model.predict([dinner_input])
		dinner_prediction = input_decode(dinner_result,0)	
		
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_id=?",(session['uid'],))
				data = cur.fetchone()
				calorie = data[17]
				protein = data[14]
				carb = data[15]
				fat = data[16]

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

		bf_cal = int(int(calorie)* 0.30)
		snack_cal = int(int(calorie)* 0.10)
		lunch_cal = int(int(calorie)* 0.35)
		dinner_cal = int(int(calorie)* 0.25)

		bf_protein = int(int(protein)* 0.30)
		snack_protein = int(int(protein)* 0.10)
		lunch_protein = int(int(protein)* 0.35)
		dinner_protein = int(int(protein)* 0.25)
		
		bf_carb = int(int(carb)* 0.30)
		snack_carb = int(int(carb)* 0.10)
		lunch_carb = int(int(carb)* 0.35)
		dinner_carb = int(int(carb)* 0.25)

		bf_fat = int(int(fat)* 0.30)
		snack_fat = int(int(fat)* 0.10)
		lunch_fat = int(int(fat)* 0.35)
		dinner_fat = int(int(fat)* 0.25)

		return render_template('recommendation.html',bf_prediction = bf_prediction[0],
													 lunch_prediction = lunch_prediction[0],
													 snack_prediction = snack_prediction[0],
													 dinner_prediction = dinner_prediction[0],
													 bf_cal = bf_cal,
													 snack_cal = snack_cal,
													 lunch_cal = lunch_cal,
													 dinner_cal = dinner_cal,
													 bf_protein = bf_protein,
													 snack_protein = snack_protein,
													 lunch_protein = lunch_protein,
													 dinner_protein = dinner_protein,
													 bf_carb = bf_carb,
													 snack_carb = snack_carb,
													 lunch_carb = lunch_carb,
													 dinner_carb = dinner_carb,
													 bf_fat = bf_fat,
													 snack_fat = snack_fat,
													 lunch_fat = lunch_fat,
													 dinner_fat = dinner_fat,

													 )

@app.route("/recommend_setup", methods = ['GET','POST'])
def recommend_setup():
	if request.method == 'GET':
		print(session['u_info'][12])
		return render_template('recommendsetup.html')

	else:
		
		return render_template('recommendsetup.html')

@app.route("/diabetes_detection", methods = ['GET','POST'])
def diabetes_detection():
	if request.method == 'GET':
		return render_template('diabetes_detection.html')
	else:
		try:
			# Get form data
			pregnancies = float(request.form['pregnancies'])
			glucose = float(request.form['glucose'])
			blood_pressure = float(request.form['blood_pressure'])
			skin_thickness = float(request.form['skin_thickness'])
			insulin = float(request.form['insulin'])
			bmi = float(request.form['bmi'])
			diabetes_pedigree = float(request.form['diabetes_pedigree'])
			age = float(request.form['age'])
			
			# Load model and scaler
			model = pickle.load(open('diabetes_model.pkl', 'rb'))
			scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
			
			# Prepare input data
			input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
									insulin, bmi, diabetes_pedigree, age]])
			
			# Scale the input
			input_scaled = scaler.transform(input_data)
			
			# Make prediction
			prediction = model.predict(input_scaled)[0]
			prediction_proba = model.predict_proba(input_scaled)[0]
			
			# Prepare result
			result = "Positive" if prediction == 1 else "Negative"
			confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
			confidence_percent = round(confidence * 100, 2)
			
			return render_template('diabetes_detection.html', 
								 result=result, 
								 confidence=confidence_percent,
								 show_result=True,
								 pregnancies=pregnancies,
								 glucose=glucose,
								 blood_pressure=blood_pressure,
								 skin_thickness=skin_thickness,
								 insulin=insulin,
								 bmi=bmi,
								 diabetes_pedigree=diabetes_pedigree,
								 age=age)
		except Exception as e:
			flash(f'Error in prediction: {str(e)}')
			return render_template('diabetes_detection.html', error=str(e))

@app.route("/index", methods = ['GET','POST'])
def index():
	if request.method == 'GET':

		try:	
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_id=?",(session['uid'],))
				u_data = cur.fetchone()
				weight = u_data[6]

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('index.html', weight = weight)

	else:
		getweight = request.form['weight']
		try:
			with get_connection() as conn:
				cur = conn.cursor()
			cur.execute("update user set u_weight=? where u_id=?",(getweight,session['uid'],))
			conn.commit()

			cur.execute("select * from user where u_id=?",(session['uid'],))
			u_data = cur.fetchone()
			weight = u_data[6]

				

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('index.html',weight = weight)

@app.route("/about", methods = ['GET','POST'])
def about():
	if request.method == 'GET':
		return render_template('about.html')

	else:
		return render_template('about.html')

@app.route('/logout')
def logout():
	session.pop('uid',None)
	session.pop('u_pass',None)
	session.pop('u_info',None)
	flash('You have successfully logged out')
	return redirect(url_for('login'))

if __name__=="__main__":
	app.run(port=5000,debug="true")
