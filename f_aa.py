from __future__ import division     # allows for Python 2 & 3 compatibility
import numpy as np                  # import numpy


def f_aa( freq, T, RH, p=101325):
    '''
    F_AA    Calculate Atmospheric Attenuation (dB/m)
    
    Calculate the atmospheric attenuation as a function of sound frequency,
    temperature, relative humidity and pressure and frequency, based on ISO
    9613 and Rossing (2007)
    
    SYNTAX: AA = f_aa(freq, T, RH [,p])
    
    INPUT:
    1) freq: frequency (Hz)
    2) T: temperature (degree C)
    3) RH: relative humidity (%)
    4) p (optional): pressure (Pa) (standard value: 101325)
    
    3 or 4 inputs can be entered. If p is not specified, its standard value
    will be used (101325 Pa)
    SIZE of all inputs: scalars or vectors. All vectors must be of the same
    size (1xN or Nx1). Vectors must be specified as an array-like object.
    
    OUTPUT:
    1) atmatt: atmospheric attenuation (dB/m)
    
    Calculations are based on the ISO Standard 9613. The same formula are
    presented in Rossing (2007): Springer Handbook of Acoustics. Chapter 4,
    p. 113-148. Also see http://forum.studiotips.com/viewtopic.php?t=158
    for calculations.
    
    (c) Holger R. Goerlitz, 24.11.2017, Version 1.0

    This is accompanying material of the manuscript:
    Goerlitz HR (2018): Weather conditions determine attenuation and speed
    of sound: environmental limitations for monitoring and analysing bat
    echolocation. ECOLOGY AND EVOLUTION.

    '''

    # convert into numpy arrays:
    freq = np.array(freq,dtype='float64')   # datatype: double precision float, 64 bit
    T = np.array(T,dtype='float64')
    RH = np.array(RH,dtype='float64')
    p = np.array(p,dtype='float64')


    # convert parameters:
    Ta = T+273.15       # convert to Kelvin
    Tr = Ta/293.15      # convert to relative air temperature (re 20 deg C)
    pr = p/101325       # convert to relative pressure


    # calculations:

    # Saturation Concentration of water vapor.
    # NOTE the *ERROR* in Rossing 2007!! Instead of Tref = 293.15 K (20 deg C),
    # here the triple-point isotherm temperature (273.15 K + 0.01 K =
    # 273.16 K) has to be used!
    # See ISO 9613 and http://forum.studiotips.com/viewtopic.php?t=158
    C = 4.6151 - 6.8346 * ((273.16/Ta)**1.261)

    # percentage molar concentration of water vapor:
    h = RH * 10**C / pr

    # relaxation frequencies associated with the vibration of oxygen and nitrogen:
    frO = pr * ( 24+4.04e4 * h * (0.02+h) / (0.391+h) )
    frN = pr * (Tr**(-0.5)) * (9+280*h*np.exp(-4.17*((Tr**(-1/3))-1)))

    # attenuation coefficient (Np/(m*atm)):
    alpha = \
        freq * freq \
        * ( 1.84e-11 * (1/pr) * np.sqrt(Tr) \
        + (Tr**(-2.5)) * ( 0.01275 * (np.exp(-2239.1/Ta)*1/(frO+freq*freq/frO)) \
        + 0.1068*(np.exp(-3352/Ta)*1/(frN+freq*freq/frN)) ) )

    AA = 8.686 * alpha         # convert to dB (lg(x/x0)) from Neper (ln(x/x0)).

    return (AA)

if __name__ == '__main__':
    print('f_aa: Test output for 45000 Hz, 23 degree C and 80 % RH with default parameter for p')
    print(f_aa(5000,20,80))

