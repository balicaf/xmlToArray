import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from locale import *
setlocale(LC_NUMERIC, '')

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

filename="/Users/racinecubix/Documents/helmet5.fcpxml"#path of your noisy Final Cut XML
filenameCopy="/Users/racinecubix/Documents/helmet5Copy.fcpxml"#path of your smoothed generated XML
tree = ET.parse(filename)
root = tree.getroot()
myArray=[]
print(root.text)
for x in root.findall('keyframeAnimation'):
    myArray.append(x.text)

print(myArray)
print(root.findall('library'))



def find_rec(node, element):
    for item in node.findall(element):
        yield item
        for child in find_rec(item, element):
            yield child
print(10)
list(find_rec(tree, 'library'))
print(tree.findtext('location'))
print(root.find(".//keyframe").tag)
print(root.findall(".//keyframe[@time]")[0].tag)
print(11)
"""
i = 0
#parent = ET.fromstring(filename)
for keyframe in root.findall('.//keyframe[@time]'):
    i += 1
#    print (keyframe)
#print [child.text for child in parent.findall('.//child[@attr="test"]')]
    print(root[1][0][0][0][0][0][0][0][0][i].attrib)
"""
rScaleX = []
rScaleY = []
for i in range (0, 182):#size of first dataset
    scale=(root[1][0][0][0][0][0][0][2][0][i].attrib['value'])
    scaleX=scale.split(" ")[0]
    scaleY = scale.split(" ")[1]
    scaleX=atof(scaleX)
    scaleY = atof(scaleY)
    rScaleX.append(scaleX)
    rScaleY.append(scaleY)
    print(scaleY,scaleX)


#b=atof(rScaleX[100])
print(10*rScaleY[1])
#print(rScaleX[100])


#plt.plot(rScaleY)
#plt.show()



#x = np.linspace(0,2*np.pi,100)
#y = np.sin(x) + np.random.random(100) * 0.2
y = np.asarray(rScaleY)
yScaleY = savitzky_golay(y, 35, 2) # window size 51, polynomial order 3
x = np.asarray(rScaleX)
xScaleX = savitzky_golay(x, 35, 2) # window size 51, polynomial order 3
#print(y)
"""
plt.plot(y)
plt.plot(yScaleY)
plt.show()
print(yScaleY[1])
"""
for i in range (182):
    tmp1=root[1][0][0][0][0][0][0][2][0][i]
    tmp1.set('value', str(xScaleX[i])+' '+str(yScaleY[i]))

tree.write(filenameCopy)
print(yScaleY[0])


######################
#first video second not scale but x y location
rScaleX = []
rScaleY = []
for i in range (0, 182):
    scale=(root[1][0][0][0][0][0][0][0][0][i].attrib['value'])
    scaleX=scale.split(" ")[0]
    scaleY = scale.split(" ")[1]
    scaleX=atof(scaleX)
    scaleY = atof(scaleY)
    rScaleX.append(scaleX)
    rScaleY.append(scaleY)
    print(scaleY,scaleX)

y = np.asarray(rScaleY)
yScaleY = savitzky_golay(y, 21, 3)  # window size 51, polynomial order 3
x = np.asarray(rScaleX)
xScaleX = savitzky_golay(x, 21, 3)  # window size 51, polynomial order 3

for i in range(182):
    tmp1 = root[1][0][0][0][0][0][0][0][0][i]
    tmp1.set('value', str(xScaleX[i]) + ' ' + str(yScaleY[i]))

tree.write(filenameCopy)
#"""
plt.plot(x)
plt.plot(xScaleX)
plt.show()
print(yScaleY[1])
#"""
print(yScaleY[0])
#########################
#second video scale

rScaleX = []
rScaleY = []
for i in range (0, 69):#sizeof second dataset

    scale=(root[1][0][0][0][0][1][0][2][0][i].attrib['value'])
    scaleX=scale.split(" ")[0]
    scaleY = scale.split(" ")[1]
    scaleX=atof(scaleX)
    scaleY = atof(scaleY)
    rScaleX.append(scaleX)
    rScaleY.append(scaleY)
    print(scaleY,scaleX)
y = np.asarray(rScaleY)
yScaleY = savitzky_golay(y, 35, 2) # window size 51, polynomial order 3
x = np.asarray(rScaleX)
xScaleX = savitzky_golay(x, 35, 2) # window size 51, polynomial order 3


for i in range (69):
    print(i)
    tmp1=root[1][0][0][0][0][1][0][2][0][i]
    tmp1.set('value', str(xScaleX[i])+' '+str(yScaleY[i]))

tree.write(filenameCopy)
print(yScaleY[0])


######################
#second not scale but x y location
rScaleX = []
rScaleY = []
for i in range (0, 69):
    scale=(root[1][0][0][0][0][1][0][0][0][i].attrib['value'])
    scaleX=scale.split(" ")[0]
    scaleY = scale.split(" ")[1]
    scaleX=atof(scaleX)
    scaleY = atof(scaleY)
    rScaleX.append(scaleX)
    rScaleY.append(scaleY)
    print(scaleY,scaleX)

y = np.asarray(rScaleY)
yScaleY = savitzky_golay(y, 21, 3)  # window size 51, polynomial order 3
x = np.asarray(rScaleX)
xScaleX = savitzky_golay(x, 21, 3)  # window size 51, polynomial order 3

for i in range(69):
    tmp1 = root[1][0][0][0][0][1][0][0][0][i]
    tmp1.set('value', str(xScaleX[i]) + ' ' + str(yScaleY[i]))

tree.write(filenameCopy)
print(yScaleY[0])
