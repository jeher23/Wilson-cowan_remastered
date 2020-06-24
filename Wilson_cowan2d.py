from __future__ import division

import sys
import math
import numpy as np
import matplotlib.pylab as plt
import argparse
from scipy import integrate
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

re = 1
ri = 1
ke = 1
ki = 1
t = np.linspace(0, 100, 1000)
S0 = [0.291,0.033] #initial condition
tau = 1


def transduction_function(x, mu, a):
    return (1 / (1 + np.exp(-a * (x - mu))))


def inverse_transduction_function(x1,x2,wij,w22, mu, a,):
    alpha=(1/x2)-1
    var3=np.log(alpha-1)/a
    final =mu-var3 -w22*x2
    return final/wij

parser = argparse.ArgumentParser(description='Parameters for the Wilson and Cowan Simulation')
parser.add_argument('-w11', type=float, dest='w11', help='weight between Excitatory - Excitatory layers')
parser.add_argument('-w12', type=float, dest='w12', help='weight between Excitatory - Inhibitory layers')
parser.add_argument('-w21', type=float, dest='w21', help='weight between Excitatory - Inhibitory layers')
parser.add_argument('-w22', type=float, dest='w22', help='weight between Inhibitory - Inhibitory layers')
parser.add_argument('-a1', type=float, dest='a1', help='Maximum slope value for the excitatory population')
parser.add_argument('-a2', type=float, dest='a2', help='Maximum slope value for the inhibitory population')
parser.add_argument('-mu_1', type=float, dest='mu_1', help='Membrane time constant for excitatory population')
parser.add_argument('-mu_2', type=float, dest='mu_2', help='Membrane time constant for inhibitory population')
args = parser.parse_args()

# making the phase portra2t graph to see the stability of the steady states.
def phase_portrait(nam):
    xval, yval = np.meshgrid(np.arange(-0.301,1.3 , 0.04),np.arange(-0.301, 1.3, 0.04))
    val1 = args.w11*xval + args.w12*yval
    xdot = -xval + (ke-re*xval)*transduction_function(val1,args.mu_1, args.a1)
    val2 = args.w21*xval + args.w22*yval
    ydot = -yval + (ke-re*yval)*transduction_function(val2,args.mu_2, args.a2)
    fig2,ax2 = plt.subplots()
    fig2.set_size_inches((11, 8), forward=False)
    testx = np.arange(0.001,0.3 , 0.0004)
    testy =  np.arange(0.001,0.3 , 0.0004)
    x2_nucl =  inverse_transduction_function(testx,testx,args.w12,args.w11,args.mu_1,args.a1)
    x1_nucl = inverse_transduction_function(testy,testy,args.w21,args.w22,args.mu_2,args.a2)
    ax2.streamplot(xval, yval, xdot, ydot)
    ax2.plot( testx,x2_nucl, label=r'$\frac{dx_2}{dt}=0$', linestyle='-.')
    ax2.plot(x1_nucl,testy, label=r'$\frac{dx_1}{dt}=0$', linestyle='-.')
    #ax2.set_title("phase portrait")
    ax2.set_xlabel(r"${x_1}$")
    ax2.set_ylabel(r"${x_2}$")
    ax2.grid()
    plt.legend(loc='best')
    fig2.savefig("img/portait_test{}.png".format(nam),dpi=1000,orientation='portrait',quality=95)
    plt.show()

phase_portrait("test")

# making the graph of the systeme
def Sys(X, t):
    val1 = args.w11*X[0] + args.w12*X[1]
    val2 = args.w21*X[0] + args.w22*X[1]
    # here X[0] = x and x[1] = y
    return np.array([ -X[0] +(1-X[0])*transduction_function(val1,args.mu_1, args.a1),  -X[1] +(1-X[1])*transduction_function(val2,args.mu_2, args.a2)])
X, infodict = integrate.odeint(Sys, S0, t,full_output=True)
x,y = X.T
fig,ax = plt.subplots()
fig.set_size_inches((11, 8), forward=False)
ax.plot(t,x, '-',label=r"$x_1$")
ax.plot(t,y, '--',label=r"$x_2$")
ax.set_xlabel("time")
ax.set_ylabel("neurons activity")
ax.grid(True)
ax.legend(loc='best')
fig.savefig("img/systeme.png",dpi=1000,orientation='portra2t',quality=95)
plt.show()

#python Wilson_cowan2d.py -w11 16 -w12 -12 -w21 15 -w22 -3 -a1 1.3 -mu_1 2.5 -a2 2 -mu_2 3.7
