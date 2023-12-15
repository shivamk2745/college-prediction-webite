from flask import Flask, render_template, request
from CollegeAllotment import algorithms
from CollegeAllotment import college_links
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')


@app.route('/predictCollege')
def predictCollege():
    return render_template("predictCollege.html")

@app.route('/results', methods=['GET', 'POST'])
@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        marks = request.form['marks']
        algorithm = request.form['algo']
        caste = request.form['caste']
        c = algorithms()

        if algorithm == "KNN":
            college_info = c.predictKNN_with_links(marks, caste)
            return render_template("results.html", colleges=[college_info])
        elif algorithm == "SVM":
            college = c.predictSVM(marks, caste)
            college_info = {"name": college, "cutoff": "NA", "url": college_links.get(college, 'Website not found')}
            return render_template("results.html", colleges=[college_info])

    return render_template("results.html", colleges=None)




@app.route('/colleges')
def colleges():
    return render_template("colleges.html")

@app.route('/sortedResults')
def sortedResults():
    start = float(request.args.get('minpercentage'))
    end = float(request.args.get('maxpercentage'))
    results = int(request.args.get('results'))
    caste = str(request.args.get('caste'))
    c = algorithms()
    my_list = c.get_by_range(start, end, results, caste)
    # return render_template("sortedResults.html", my_list=my_list, size=len(my_list))
    return render_template("sortedResults.html", my_list=my_list, size=len(my_list), college_links=college_links)
if __name__ == '__main__':
    app.debug = True
    app.run()


