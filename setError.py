def setError(data_fuzzy):

    a = 8
    b = 25
    c = 4
    d = 45
    e = 10
    f = 17
    g = 35

    m1 = data_fuzzy[0]
    m2 = data_fuzzy[1]
    m3 = data_fuzzy[2]
    de1 = data_fuzzy[3]
    de2 = data_fuzzy[4]
    de3 = data_fuzzy[5]
    p1 = data_fuzzy[6]
    p2 = data_fuzzy[7]
    p3 = data_fuzzy[8]
    q1 = data_fuzzy[9]
    q2 = data_fuzzy[10]
    q3 = data_fuzzy[11]

    x_array = []
    y_array = []

    error = 0.0

    for i in range(1, 1000):

        x = i/10
        x_array.append(x)

        mf1 = math.exp((-math.pow((x-m1), 2))/(2*math.pow(de1, 2)))
        mf2 = math.exp((-math.pow((x-m2), 2))/(2*math.pow(de2, 2)))
        mf3 = math.exp((-math.pow((x-m3), 2))/(2*math.pow(de3, 2)))

        bf = mf1+mf2+mf3
        a1 = mf1*(p1*x+q1)
        a2 = mf2*(p2*x+q2)
        a3 = mf3*(p3*x+q3)
        af = a1+a2+a3

        y = af/bf
        y_array.append(y)

        # Correct curve point
        correct_curve_point = (
            a * (b * math.sin(x/c)+(d*math.cos(x/e)))) + (f*x) - g

        # operations are correct, now create the comparation and add results to error
        error += abs(correct_curve_point - y)

    return error