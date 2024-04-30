#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request,redirect,url_for
# Import your face capture and face detection scripts
from capture_face import capture_faces
from detect_face import detect_faces

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_faces', methods=['POST'])
def capture():
    name = request.form['name']
    # Pass the provided name to the face capture function
    capture_faces(name)
    return redirect(url_for('index'))

@app.route('/detect_faces')
def detect():
    # Call the face detection function
    detect_faces()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




