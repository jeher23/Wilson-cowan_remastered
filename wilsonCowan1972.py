from __future__ import division

import sys
import math
import numpy as np
import matplotlib.pylab as plt
import argparse
from scipy import integrate

def transduction_function(x, theta, a):
    return (1 / (1 + np.exp(-a * (x - theta)))) #- (1 / (1 + np.exp(a * (theta))))


def inverse_transduction_function(x1,x2,wij,wii, theta, a,p):
    #print(x, math.log((1-x)/x))
    var1= x2/(1-x2)
    var2 = 0#1/(1+np.exp(a*theta))
    alpha=(1/x2)-1
    #print(alpha)
    var3=np.log(alpha-1)/a

    final =theta-var3 -wii*x2 - p
    return final/wij

parser = argparse.ArgumentParser(description='Parameters for the Wilson and Cowan Simulation')
parser.add_argument('-wee', type=int, dest='wee', help='Weight between Excitatory - Excitatory layers')
parser.add_argument('-wei', type=int, dest='wei', help='Weight between Excitatory - Inhibitory layers')
parser.add_argument('-wie', type=int, dest='wie', help='Weight between Excitatory - Inhibitory layers')
parser.add_argument('-wii', type=int, dest='wii', help='Weight between Inhibitory - Inhibitory layers')
parser.add_argument('-ae', type=float, dest='ae', help='Maximum slope value for the excitatory population')
parser.add_argument('-ai', type=float, dest='ai', help='Maximum slope value for the inhibitory population')
parser.add_argument('-theta_e', type=float, dest='theta_e', help='Membrane time constant for excitatory population')
parser.add_argument('-theta_i', type=float, dest='theta_i', help='Membrane time constant for inhibitory population')
args = parser.parse_args()

re = 1
ri = 1
ke = 1
ki = 1
q = 0
p = 2

print('end')
def phase_portrait(nam):
    xval, yval = np.meshgrid(np.arange(-0.301,1.3 , 0.04),np.arange(-0.301,1.3 , 0.04))
    val1 = args.wee*xval + args.wei*yval + p
    xdot = -xval + (ke-re*xval)*transduction_function(val1,args.theta_e, args.ae)
    val2 = args.wie*xval + args.wii*yval + q
    ydot = -yval + (ke-re*yval)*transduction_function(val2,args.theta_i, args.ai)
    fig2,ax2 = plt.subplots()
    fig2.set_size_inches((11, 8), forward=False)
    x2_nucl = inverse_transduction_function(yval,xval,args.wei,args.wee,args.theta_e,args.ae,p)
    x1_nucl = inverse_transduction_function(xval,yval,args.wie,args.wii,args.theta_i,args.ai,q)
    testx = np.arange(0.001,0.3 , 0.0004)
    testy =  np.arange(0.001,0.3 , 0.0004)
    b =  inverse_transduction_function(testx,testx,args.wei,args.wee,args.theta_e,args.ae,p)
    c = inverse_transduction_function(testy,testy,args.wie,args.wii,args.theta_i,args.ai,q)
    #print(np.shape(xre),np.shape(yre),np.shape(b),np.shape(c))

    ax2.streamplot(xval, yval, xdot, ydot)
    ax2.plot( testx,b, label=r'$\frac{dI}{dt}=0$', linestyle='-.')
    ax2.plot(c,testy, label=r'$\frac{dE}{dt}=0$', linestyle='-.')
    ax2.set_title("phase portrait")
    ax2.set_xlabel(r"${E}$")
    ax2.set_ylabel(r"${I}$")
    ax2.grid()
    plt.legend(loc='best')
    #fig2.savefig("img/portait_test{}.png".format(nam),dpi=1000,orientation='portrait',quality=95)
    plt.show()

phase_portrait("test")





def Sys(X, t):
    val1 = args.wee*X[0] + args.wei*X[1] + p
    val2 = args.wie*X[0] + args.wii*X[1] + q
    # here X[0] = x and x[1] = y
    return np.array([ -X[0] +(1-X[0])*transduction_function(val1,args.theta_e, args.ae),  -X[1] +(1-X[1])*transduction_function(val2,args.theta_i, args.ai)])
a, b = 0, 10
t = np.linspace(a, b, 1000)
S0 = [0.291,0.33]
X, infodict = integrate.odeint(Sys, S0, t,full_output=True)
x,y = X.T
fig,ax = plt.subplots()
fig.set_size_inches((11, 8), forward=False)
ax.plot(t,x, '-')
ax.plot(t,y, '--')
#phase_portrait("Test_phase_portrait")
ax.set_xlabel("time")
ax.set_ylabel("neurons activity")
ax.grid()
ax.legend(loc='best')
plt.show()
