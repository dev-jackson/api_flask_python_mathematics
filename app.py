from flask import Flask,jsonify,request,render_template, redirect, url_for
import statistics as st
import numpy as npy
import json as js
app = Flask(__name__,template_folder='template')

@app.route('/home/statistics/min_squares', methods=['POST'])
def page1():
    array = []
    for i in range(0,len(request.json['result'])):
        array.append(request.json['result'][i.__str__()])
    res = st.min_square(array,request.json['cant']['val'])
    formatResult=[] 
    for i in range(len(res)):
        formatResult.append({i:float("{0:.2f}".format(res[i]))})
    return jsonify({"result":formatResult})

@app.route('/home/statistics/northwest_corner')
def northwest_corner():
    test ={
        "0":{
            "0":5,
            "1":2,
            "2":7,
            "3":3,
            "4":80
            },
        "1":{
            "0":3,
            "1":6,
            "2":6,
            "3":1,
            "4":30
            },
        "2":{
            "0":6,
            "1":1,
            "2":2,
            "3":4,
            "4":60
            },
         "3":{
            "0":4,
            "1":3,
            "2":6,
            "3":6,
            "4":45
            },
         "4":{
            "0":70,
            "1":40,
            "2":70,
            "3":35,
            }
        }
    #res = mc.northwest_corner(test)
    demand = npy.array([70, 40, 70, 35])
    supply = npy.array([80, 30, 60, 45])

    costs = npy.array([[5., 2., 7., 3.],
                      [3., 6., 6., 1.],
                      [6., 1., 2., 4.],
                      [4., 3., 6., 6.]])
    res = st.northwest_corner(costs,supply,demand)
    print(res)
    return "s"
@app.route('/home/statistics/mean')
def mean():
    res = st.mean([2, 3, 6, 8, 11])
    return jsonify({"result": res})
@app.route('/home/statistics/median')
def median():
    res = st.median([2, 3, 4, 4, 5, 5, 5, 6, 6])
    return jsonify({"result": res})
@app.route('/home/statistics/typical_deviation')
def typical_deviation():
    mean = st.mean([2, 3, 6, 8, 11])
    res = st.typical_deviation(mean,[2, 3, 6, 8, 11])
    return jsonify({"result": float("{0:.2f}".format(res))})
@app.route('/home/statistics/typical_deviation_table/<result>')
def view_typical_deviation_table(result):
    return render_template('typical_deviation_table.html',res=result)
@app.route('/home/statistics/typical_deviation_table',methods=['GET','POST'])
def typical_deviation_table():
    if request.method == "POST":
        x_i = request.json['res']['x_i']
        f_i = request.json["res"]['f_i']
    res = st.typical_deviation_table([x_i,f_i])
    data = { 'init_table':{
                'x_i':x_i,
                'f_i':f_i
            },
            'table_result':[res[0]],
             'mean': res[1],
             'deviation':res[2]}
    return data;

@app.route('/home/statistics/typical_deviation_interval')
def typical_deviation_interval():  
    return render_template('typical_deviation.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)
