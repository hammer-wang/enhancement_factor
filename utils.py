from tmm import coh_tmm
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import pi

DATABASE = './data'

class TMM_sim():
    def __init__(self, mats=['Ge'], wavelength=np.arange(0.38, 0.805, 0.01), substrate='Cr', substrate_thick=500, database=None):
        '''
        This class returns the spectrum given the designed structures.
        '''

        if database:
            self.database = database
        else:
            self.database = DATABASE

        self.mats = mats
        # include substrate
        self.all_mats = mats + [substrate] if substrate not in ['Glass', 'Air'] else mats
        self.wavelength = wavelength
        self.nk_dict = self.load_materials()
        self.substrate = substrate
        self.substrate_thick = substrate_thick

    def load_materials(self):
        '''
        Load material nk and return corresponding interpolators.

        Return:
            nk_dict: dict, key -- material name, value: n, k in the 
            self.wavelength range
        '''
        nk_dict = {}

        for mat in self.all_mats:
            nk = pd.read_csv(os.path.join(self.database, mat + '.csv'))
            nk.dropna(inplace=True)
            wl = nk['wl'].to_numpy()
            index = (nk['n'] + nk['k'] * 1.j).to_numpy()
            mat_nk_data = np.hstack((wl[:, np.newaxis], index[:, np.newaxis]))


            mat_nk_fn = interp1d(
                    mat_nk_data[:, 0].real, mat_nk_data[:, 1], kind='quadratic')
            # print(self.wavelength)
            nk_dict[mat] = mat_nk_fn(self.wavelength)

        return nk_dict

    def spectrum(self, materials, thickness, theta=0, plot=False, title=False):
        '''
        Input:
            materials: list
            thickness: list
            theta: degree, the incidence angle

        Return:
            s: array, spectrum
        '''
        degree = pi/180
        if self.substrate != 'Air':
            thickness.insert(-1, self.substrate_thick) # substrate thickness

        R, T, A = [], [], []
        for i, lambda_vac in enumerate(self.wavelength * 1e3):

            # we assume the last layer is glass
            if self.substrate == 'Glass':
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1.45, 1]
            elif self.substrate == 'Air':
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [1]
            else:
                n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict[self.substrate][i], 1]

            # n_list = [1] + [self.nk_dict[mat][i] for mat in materials] + [self.nk_dict['Cr'][i]]

            # mport pdb; pdb.set_trace()
            res = coh_tmm('s', n_list, thickness, theta * degree, lambda_vac)
            if theta != 0:
                res_t = coh_tmm('p', n_list, thickness, theta * degree, lambda_vac)
                res['R'] = (res['R'] + res_t['R']) / 2
                res['T'] = (res['T'] + res_t['T']) / 2
            R.append(res['R'])
            T.append(res['T'])

        R, T = np.array(R), np.array(T)
        A = 1 - R - T

        if plot:
            self.plot_spectrum(R, T, A)
            if title:
                thick = thickness[1:-1]
                title = ' | '.join(['{}nm {}'.format(d, m)
                                    for d, m in zip(thick, materials)])
                if self.substrate is not 'Air':
                    title = 'Air | ' + title + ' | {}nm {} '.format(self.substrate_thick, self.substrate) + '| Air'
                else:
                    title = 'Air | ' + title + ' | Air'
                plt.title(title, **{'size': '10'})
            # plt.show()

        return R, T, A

    def plot_spectrum(self, R, T, A):

        plt.plot(self.wavelength, R, self.wavelength, T, self.wavelength, A, linewidth=3)
        plt.ylabel('R/T/A')
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.legend(['R: Average = {:.2f}%'.
                    format(np.mean(R)*100),
                    'T: Average = {:.2f}%'.
                    format(np.mean(T)*100),
                    'A: Average = {:.2f}%'.
                    format(np.mean(A)*100)])
        plt.grid('on', linestyle='--')
        plt.ylim([0, 1])