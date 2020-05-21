import os,sys,time
import multiprocessing
import numpy as np
from copy import copy
from shutil import copy as cp
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from progress.bar import FillingCirclesBar

def Pool_Processes(fcn,args,processes=1):
    pool = multiprocessing.Pool(processes)
    out = pool.map(fcn,args)  
    pool.close()
    pool.join()
    del pool
    return out

def get_random_seeing_manga(seed=None,seeing_pool=None):
    '''Select random seeing full-width at half-maximum (FWHM) [arcsec] from the set of seeing values in MaNGA. By default, the seeing values derive from the MaNGA drpall tables which include min, max, and median guide star seeing estimates for every combined set of exposures.
    
    Keywords:
    
        * `seed` (int) is the random seed for seeing selection. Setting the seed allows one to reproduce the selection.
    
        * `seeing_pool` (list or np.ndarray) is a manually generated pool of seeing FWHM [arcesec] from which to draw.
        
    Returns: seeing FWHM [arcsec].'''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
        
    if seeing_pool is None:
        drpall_name = 'drpall-v2_4_3.fits'
        # get drp all file for guide-star seeing
        if not os.access(drpall_name,0): 
            drpall_backup = '/home/bottrell/scratch/Merger_Kinematics/RealSim-IFS/Resources/{}'.format(drpall_name)
            # check for local copy in Resources directory
            if os.access(drpall_backup,0):
                cp(drpall_backup,drpall_name)
            else:
                drpall_url = 'https://data.sdss.org/sas/dr16/manga/spectro/redux/v2_4_3/{}'.format(drpall_name)
                os.system('wget {}'.format(drpall_url))
        drpall_data = fits.getdata(drpall_name)
        seeing_pool = drpall_data['SEEMED']
    else:
        if type(seeing_pool) not in [list,np.ndarray]:
            raise Exception('seeing_pool not correct instance. Must be list or 1-D numpy array. Stopping...')
    return np.random.choice(seeing_pool,replace=True)


def get_random_redshift_manga(seed=None,redshift_pool=None):
    '''Select random redshift from the set of redshifts for ALL MaNGA targets. By default, the redshifts derive from the targeting table found here: https://www.sdss.org/dr16/manga/manga-target-selection/targeting-catalog/ which are taken from the NSA catalogues. To obtain the target redshifts, I use the bitmasks for primary, secondary, and colour-enchanced (primary+) targets.
    
    Keywords:
    
        * `seed` (int) is the random seed for redshift selection. Setting the seed allows one to reproduce the selection.
        
        * `redshift_pool` (list or np.ndarray) is a manually generated pool of redshifts from which to draw.
        
    Returns: redshift, z.'''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
    if redshift_pool is None:
        zcat_name = 'MaNGA_target_redshifts-all.txt'
        if not os.access(zcat_name,0):   
            # check for local copy in Resources directory
            zcat_backup = '/home/bottrell/scratch/Merger_Kinematics/RealSim-IFS/Resources/{}'.format(zcat_name)
            if os.access(zcat_backup,0):
                cp(zcat_backup,zcat_name)
                redshift_pool = np.loadtxt(zcat_name)
            else:
                # create table from targeting catalog
                targetcat_name = 'MaNGA_targets_extNSA_tiled_ancillary.fits'
                if not os.access(targetcat_name,0):
                    targetcat_url = 'https://data.sdss.org/sas/dr16/manga/target/v1_2_27/{}'.format(targetcat_name)
                    os.system('wget {}'.format(targetcat_url))
                redshift_pool = fits.getdata(targetcat_name)['NSA_Z']
                bitmask = fits.getdata(targetcat_name)['MANGA_TARGET1']
                pri_mask = (bitmask & 1024)!=0 # primary
                sec_mask = (bitmask & 2048)!=0 # secondary
                cen_mask = (bitmask & 4096)!=0 # colour-enhanced (primary+)
                redshift_pool = redshift_pool[(pri_mask+sec_mask+cen_mask)>=1]
        else:
            redshift_pool = np.loadtxt(zcat_name) 
    else:
        if type(redshift_pool) not in [list,np.ndarray]:
            raise Exception('redshift_pool not correct instance. Must be list or 1-D numpy array. Stopping...')
    return np.random.choice(redshift_pool,replace=True)


def apply_seeing(datacube, kpc_per_pixel,  redshift = 0.05, seeing_model='manga', 
                 seeing_fwhm_arcsec=1.5, cosmo=None, use_threading=False, n_threads=1):
    '''Apply atmospheric or other pre-instrument seeing model to datacube corresponding to the target redshift. Currently, this function only allows a fixed seeing that is applied to all wavelength/velocity elements of the input datacube. However, the function can be easily modified to allow a seeing model which varies with wavelength. The seeing angular full width at half-maximum (FWHM) [arcsec] can be converted to pixels in the datacube using the redshift, z, and the physical spatial pixel scale of the datacube.
    
    Keywords:
    
        * `datacube` (np.ndarray) is the datacube to be convolved slice-by-slice with the seeing. The slices can either be wavelength or line-of-sight velocity. The wavelength and velocity channels should be first in the axis order (i.e. (Nels,spatial_y,spatial_x), where Nels is the number of wavelenght or velocity elements).
    
        * `kpc_per_pixel` (int, float) is the physical spatial pixel scale for the datacube in [kpc]. This is needed to compute the angular scale of each pixel once a given angular diameter distance is adopted.
        
        * `redshift` (int,float) the target redshift at which the source is to be observed. The redshift is used to determine the angular diameter distance to the source and subsequently compute the angular size it subtends in the sky. 
        
        * `seeing_model` (string) [options: manga (default), gaussian] is seeing model which approximates the atmospheric or pre-instrumental seeing. 
        
            - `manga` atmospheric guide-star seeing is modelled as a combination of two gaussians (private communication w/ David Law, Jan, 2020):
            
                theta = seeing_fwhm_pixels / 1.05 
                kernel = 9/13 * Gaussian(fwhm=theta) + 4/13 * Gaussian(fwhm=2*theta)
                
            - `gaussian` a basic Gaussian seeing model:
                kernel = Gaussian(fwhm=seeing)
        
        * `seeing_fwhm_arcsec` (int,float) the angular scale of the seeing in [arcsec]
        
        * `cosmo` (astropy.cosmology instance) the cosmology which defines angular scales and distances corresponding to a given redshift. Default is a LambdaCDM cosmology with Planck 2018 parameters (https://arxiv.org/pdf/1807.06209.pdf) [last updated: February, 2020].
        
        * `use_threading` (boolean) whether to use multiprocessing Pool to apply convolutions in each slice. 
        
        * `n_threads` (int) the number of threads to be used if `use_threading` is True.
        
    Returns: `outcube` (np.ndarray) Seeing-convolved datacube with same shape as input datacube.
    '''
    
    # threadable kernel-convolution function
    def convolve_slice(args):
        img,kernel=args
        return convolve(img,kernel)
    
    # cosmology
    if cosmo == None:
        cosmo=FlatLambdaCDM(H0=67.4,Om0=0.315)
    
    # speed of light [m/s]
    speed_of_light = 2.99792458e8
    # kiloparsec per arcsecond scale
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60. # [kpc/arcsec]
    # luminosity distance in Mpc (not used but added to header)
    luminosity_distance = cosmo.luminosity_distance(z=redshift).value # [Mpc]
    # compute angular pixel scale from cosmology
    arcsec_per_pixel = kpc_per_pixel / kpc_per_arcsec # [arcsec/pixel]
    
    valid_seeing_models = ['manga','gaussian']
    
    if seeing_model == 'manga':
        # MaNGA seeing model from private communication w/ David Law (January, 2020)
        from astropy.convolution import Gaussian2DKernel
        seeing_std_pixels = seeing_fwhm_arcsec/arcsec_per_pixel/1.05/2.335
        kernel = (9/13*Gaussian2DKernel(seeing_std_pixels)).__add__(4/13*Gaussian2DKernel(2*seeing_std_pixels))          
    elif seeing_model == 'gaussian':
        # standard gaussian seeing
        from astropy.convolution import Gaussian2DKernel
        seeing_std_pixels = seeing_fwhm_arcsec/arcsec_per_pixel/2.335
        kernel = Gaussian2DKernel(seeing_std_pixels)  
    else:
        raise Exception('Incompatible atmospheric seeing model. Choose seeing model from list of compatible seeing models: \n {}'.format(valid_seeing_models))
    
    # convolve, use threading or not
    if not use_threading:
        outcube = np.zeros_like(datacube)
        for i in range(len(outcube)):
            outcube[i]=convolve_slice((datacube[i],kernel))
    else:
        if type(n_processes) != int:
            raise Exception('use_threading is true but n_processes is not an integer. Stopping...')
        pool = multiprocessing.Pool(n_processes)
        args = [(datacube[i],kernel) for i in range(len(datacube))]
        outcube = np.array(pool.map(convolve_slice,args))
        pool.close()
        pool.join()
        del pool

    return outcube


######################### Staged for removal #########################
def Convolve_Slice(args):
    data,kernel=args
    return convolve(data,kernel)

def Prepare_IFS(filename,redshift=0.046,psf_type='None',psf_fwhm_arcsec=1.5,
                cosmo=FlatLambdaCDM(H0=70,Om0=0.3)):
    
    if not os.access(filename,0):
        sys.exit('Datacube not found. Quitting...')
    
    if filename.endswith('.fits'):
        data = np.transpose(fits.getdata(filename),[2,0,1])
        hdr = fits.getheader(filename)
    else:
        sys.exit('Datacube is not in FITS format. Quitting...')
        
    # speed of light [m/s]
    speed_of_light = 2.99792458e8
    # kiloparsec per arcsecond scale
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z=redshift).value/60. # [kpc/arcsec]
    # luminosity distance in Mpc
    luminosity_distance = cosmo.luminosity_distance(z=redshift) # [Mpc]
        
    fov_size_kpc = hdr['FOVSIZE']/1000.
    kpc_per_pixel = fov_size_kpc/hdr['NAXIS2']
    arcsec_per_pixel = kpc_per_pixel/kpc_per_arcsec
    
    if psf_type is not 'None':
        valid_psf_types = ['Gaussian',]
        if psf_type in valid_psf_types:
            if psf_type == 'Gaussian':
                std = psf_fwhm_arcsec/arcsec_per_pixel/2.355
                kernel = Gaussian2DKernel(x_stddev=std,y_stddev=std)  
            if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
                cpu_count = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            else:
                cpu_count = multiprocessing.cpu_count()
            args = [(data[i],kernel) for i in range(len(data))]
            data = np.array(Pool_Processes(Convolve_Slice,args,cpu_count))
        else:
            raise Exception('Choose PSF profile from list of valid PSF profiles: \n {}'.format(valid_psf_types))
    return data
######################### Staged for removal #########################

def Generate_Maps_From_File(losvd_name):
    if not os.access(losvd_name,0):
        print('LOSVD file not found')
    else:
        losvd_head = fits.getheader(losvd_name)
        losvd_data = fits.getdata(losvd_name)
        losvd_shape = losvd_data.shape
        vlim = losvd_head['VLIM'] # km/s
        delv = losvd_head['DELV'] # km/s
        fov = losvd_head['FOVSIZE']/1000. # kpc
        vel = np.linspace(-vlim,vlim,losvd_shape[-1],endpoint=False)+delv/2.
        sum_wi = np.nansum(losvd_data,axis=-1)
        sum_wivi = np.nansum(losvd_data*vel,axis=-1)
        vbar = sum_wivi/sum_wi
        Nprime = np.nansum(losvd_data>0,axis=-1)
        vstd = np.nansum(losvd_data*(vel-vbar[...,np.newaxis])**2,axis=-1)
        vstd /= (Nprime-1)/Nprime*sum_wi
        vstd = np.sqrt(vstd)
        losvd_maps = np.array([sum_wi,vbar,vstd])
        return losvd_maps
    
def Generate_Maps_From_Data(losvd_name,losvd_data):
    if not os.access(losvd_name,0):
        print('LOSVD file not found')
    else:
        losvd_head = fits.getheader(losvd_name)
        losvd_data = losvd_data.transpose(1,2,0)
        losvd_shape = losvd_data.shape
        vlim = losvd_head['VLIM'] # km/s
        delv = losvd_head['DELV'] # km/s
        fov = losvd_head['FOVSIZE']/1000. # kpc
        vel = np.linspace(-vlim,vlim,losvd_shape[-1],endpoint=False)+delv/2.
        sum_wi = np.nansum(losvd_data,axis=-1)
        sum_wivi = np.nansum(losvd_data*vel,axis=-1)
        vbar = sum_wivi/sum_wi
        Nprime = np.nansum(losvd_data>0,axis=-1)
        vstd = np.nansum(losvd_data*(vel-vbar[...,np.newaxis])**2,axis=-1)
        vstd /= (Nprime-1)/Nprime*sum_wi
        vstd = np.sqrt(vstd)
        losvd_maps = np.array([sum_wi,vbar,vstd])
        return losvd_maps   

def Err_n_observations():
    print('You must select `n_observations` that is either:')
    print('(1) `Classic` for exact MaNGA specs and 3-exposure pattern; OR')
    print('(2) An integer `n_observations` greater than zero.')
    sys.exit(0)
    
def Check_for_list(x,n_observations):
    if type(x) is np.ndarray:
        if len(x) == n_observations:
            return x
    elif type(x) is list:
        if len(x) == n_observations:
            return np.array(x)
    else:
        err_msg = ['When `n_observations` is set greater than 1 and not `Classic`, you must set up the pattern.\n',
                   'The following parameters must be all be lists or 1D numpy arrays of length `n_observations`:\n',
                   '   (1) `bundle_xoffset_arcsec`\n',
                   '   (2) `bundle_yoffset_arcsec`\n',
                   '   (3) `rotation_degrees`\n',
                   'You are seeing this error because one or more of these parameters are not in the correct format.\n']
        sys.exit(''.join(err_msg))

def Rotate(v,rotation_degrees):
    '''Rotate vectors `v` by `rotation_degrees in R2=>R2'''
    theta_rad = rotation_degrees*np.pi/180
    cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
    A_theta = np.array([[cos_theta, -sin_theta],
                        [sin_theta,  cos_theta]])
    v = np.matmul(A_theta,v)
    return v

def MaNGA_Observe(n_observations='Classic',
                  fibers_per_side=4,
                  bundle_name='None',
                  fiber_diameter_arcsec=2.480,
                  core_diameter_arcsec=1.984,
                  rotation_degrees=0.,
                  bundle_xoffset_arcsec=0.,
                  bundle_yoffset_arcsec=0.,
                  return_params=True):
    '''
    Creates a MaNGA-like IFS observing pattern. For an overview of the MaNGA observing strategy, see here: 
    https://www.sdss.org/dr14/manga/manga-survey-strategy/
    
    Or here for more rigorous descriptions:
    Law et al. (2015) https://ui.adsabs.harvard.edu/abs/2015AJ....150...19L/abstract
    Bundy et al. (2015) https://ui.adsabs.harvard.edu/abs/2015ApJ...798....7B/abstract
    
    RealSim_IFS is a public tool and you are free to use/modify it however you wish. If you use RealSim_IFS.MaNGA_Observe or an adaptation in your research, please cite the papers above which provided the technical specifications I used to create the module. I would also appreciate citation to [Bottrell et al. in prep] until the corresponding release paper is published.
    
    Keyword descriptions:
    
    * `n_observations` (int or string) is the integer number of exposures you wish to take with a given fibre bundle type. `n_observations` can also be set to `Classic` to restore all parameters to the default MaNGA instrumental specs and three-exposure offset pattern used to "dither" the data. In short, dithering is used to fill in the gaps between the fibre footprints of individual exposures. Dithering is also important because, for the MaNGA instrument, it enables adequate sampling of the typical atmospheric point-spread function (1.6 arcsec). You can alternatively experiment with your own exposure pattern and dithering strategy by setting `n_observations` to any positive integer. In the case of `n_observations` greater than 1, the `bundle_x(y)offset_arcsec` and `rotation_degrees` parameters must be lists or numpy arrays. See their descriptions for details.
    
    * `fibers_per_side` (int) is the number of fibres along each side of a MaNGA hexagonal fibre bundle. Only one `fibers_per_side` can be set for a given observation pattern (i.e. MaNGA_Observe does not allow combination of fibre bundles of different size in the same observing pattern). `fibers_per_side` must be a positive interger greater than 0. In the most limited scenario, a single fiber at (0,0) can be made by setting `n_observations` to 1 and `fibers_per_side` to 1. Setting `n_observations` to `Classic` and `fibers_per_side` to an integer greater than zero will produce the observing pattern with that particular fibre size and with the exact MaNGA core, fiber, and exposure specs.
    
    * `bundle_name` (string) supersedes `fibers_per_side`. Setting `bundle_name` to a valid MaNGA fibre bunde name (e.g. `N61`) will generate bundles with that specific fibre pattern and IGNORE the `fibers_per_side` parameter. By default, `bundle_name` is `None` and must be set to `None` to use the `fibers_per_side` keyword.
    
    * `fiber_diameter_arcsec` and `core_diameter_arcsec` (floats) are the desired angular sizes of each individual fibre (core+cladding) and core in the bundle in [arcsec]. By default, these are set to the exact MaNGA specifications.
    
    * `arcsec_per_mm` (float) [**deprecated**] is the physical-to-angular conversion ratio [arcsec/mm] at the prime-focus of instrument. For MaNGA, the reciprocal angular-to-physial scale is 60.48 microns/arcsec.
    
    * `rotation_degrees` (float) sets the counter-clockwise rotation (in degrees) of an individual exposure or full observing pattern. If `n_observations` is `Classic` or 1, `rotation_degrees` must be a single float. `n_observations` is an integer greater than 1, then `rotation_degrees` must be a list or numpy array whose elements give the rotation of each individual exposure. For the same rotation of each exposure, set the `rotation_degrees` keyword to something like np.zeros(n_observations)+rotation_degrees. 
    
    * `bundle_xoffset_arcsec` and `bundle_yoffset_arcsec` (floats) are the offsets (in arcsec) of each exposure's centre from (0,0) in x and y. If `n_observations` is `Classic` they are the default MaNGA offsets for the 3-exposure observing strategy. If `n_observations` is 1, then they must each be floats. If `n_observations` is greater than 1, both must be lists or numpy arrays in the same way as for `rotation_degrees`.
    
    * `return_params` (boolean): returns many of the variables which were used to set the observing strategy as well as some intuitively-named parameters that are computed internally when generating the output.
    
    Output:
    
    (xc_arr,yc_arr) or (xc_arr,yc_arr,params)
    
    * `xc_arr` and `yc_arr` are the coordinates of all fibers (in [arcsec]) in the `n_observations` of exposures used in the observing strategy. They each have shape (`fibers_per_bundle`, `n_observations`) where `fibers_per_bundle` is the total number of fibers in an individual bundle and is an output passed to `params`. 
    
    * `params` is the dictionary of variables that is returned if `return_params` is True.
    '''  
       
    if n_observations == 'Classic':
        # restore all params to MaNGA defaults
        fiber_diameter_arcsec=2.480
        core_diameter_arcsec=1.984
        #arcsec_per_mm=1./0.06048
        bundle_xoffset_arcsec=0.
        bundle_yoffset_arcsec=0.
        if type(rotation_degrees) is not float:
            sys.exit('When `n_observations` is `Classic`, the `rotation_degrees` must be a single float.')
        
    elif isinstance(n_observations,str):
        Error_n_observations()
        
    elif isinstance(n_observations,int) and not n_observations>0:
        Error_n_observations()
        
    elif isinstance(n_observations,int) and n_observations>1:
        rotation_degrees = Check_for_list(rotation_degrees,n_observations)
        bundle_xoffset_arcsec = Check_for_list(bundle_xoffset_arcsec,n_observations)
        bundle_yoffset_arcsec = Check_for_list(bundle_yoffset_arcsec,n_observations)
            

    # Useful quantities. See also:
    # https://www.sdss.org/dr14/manga/manga-survey-strategy/ and Law et al. (2015)
    #fiber_diameter_arcsec = fiber_diameter_mm*arcsec_per_mm # arcsec
    #core_diameter_arcsec = core_diameter_mm*arcsec_per_mm # arcsec
    cladding_arcsec = (fiber_diameter_arcsec-core_diameter_arcsec)/2. # arcsec
    exposure_offset_arcsec = fiber_diameter_arcsec/np.sqrt(3) # arscec
    valid_bundle_names = ['N7','N19','N37','N61','N91','N127']

    if bundle_name is 'None':
        fibers_per_side = fibers_per_side

    elif bundle_name in valid_bundle_names:
        if bundle_name == 'N7':
            fibers_per_side = 2
        if bundle_name == 'N19':
            fibers_per_side = 3
        if bundle_name == 'N37':
            fibers_per_side = 4
        if bundle_name == 'N61':
            fibers_per_side = 5
        if bundle_name == 'N91':
            fibers_per_side = 6
        if bundle_name == 'N127':
            fibers_per_side = 7
    else:
        print("You have not selected a valid MaNGA fiber bundle name.")
        print("Choose from the following options:")
        print([str(name) for name in valid_bundle_names])
        print('OR set the `fibers_per_side` keyword to the integer number of fibers along each edge of the desired hexagonal bundle.')

    fiber_rows_per_bundle = 2*fibers_per_side-1
    xc_arr = np.array([])
    yc_arr = np.array([])
    fiber_xoffset = 0.

    for fiber_row_index in range(fiber_rows_per_bundle):
        if fiber_row_index == 0:
            fibers_in_row = copy(fibers_per_side)
        elif fiber_row_index>=fibers_per_side:
            fibers_in_row-=1
            fiber_xoffset-=fiber_diameter_arcsec/2.
        else:
            fibers_in_row+=1
            fiber_xoffset+=fiber_diameter_arcsec/2.
        xc_new = np.arange(0,fibers_in_row)*fiber_diameter_arcsec-fiber_xoffset
        yc_new = np.zeros_like(xc_new)-fiber_row_index*np.sqrt(3)/2*fiber_diameter_arcsec
        xc_arr = np.concatenate((xc_arr,xc_new))
        yc_arr = np.concatenate((yc_arr,yc_new))

    index_center = int(len(xc_arr)/2)
    xc = xc_arr[index_center]
    yc = yc_arr[index_center]
    xc_arr -= xc
    yc_arr -= yc
    
    if n_observations == 'Classic':
        n_observations = 3
        bundle_xoffset_arcsec = np.array([0,-np.sqrt(3)/2,0])*exposure_offset_arcsec
        bundle_yoffset_arcsec = np.array([0,0.5,1])*exposure_offset_arcsec
        xc_arr = xc_arr[...,np.newaxis]+bundle_xoffset_arcsec
        yc_arr = yc_arr[...,np.newaxis]+bundle_yoffset_arcsec
        xc_dither_correction = fiber_diameter_arcsec/2/3
        yc_dither_correction = fiber_diameter_arcsec/2/np.sqrt(3)
        xc_arr+=xc_dither_correction
        yc_arr-=yc_dither_correction
        if not rotation_degrees == 0.:
            v = np.stack((xc_arr,yc_arr),axis=0)
            v_prime = np.empty_like(v)
            for i_obs in range(n_observations):
                v_prime[...,i_obs] = Rotate(v[...,i_obs],rotation_degrees)
            xc_arr = v_prime[0,:]
            yc_arr = v_prime[1,:]
    else:
        xc_arr = xc_arr[...,np.newaxis]+np.zeros(n_observations)
        yc_arr = yc_arr[...,np.newaxis]+np.zeros(n_observations)
        v = np.stack((xc_arr,yc_arr),axis=0)
        v_prime = np.empty_like(v)
        if n_observations == 1:
            v_prime[...,0] = Rotate(v[...,0],rotation_degrees)
        else:    
            for i_obs in range(n_observations):
                v_prime[...,i_obs] = Rotate(v[...,i_obs],rotation_degrees[i_obs])
        xc_arr = v_prime[0,:]+bundle_xoffset_arcsec
        yc_arr = v_prime[1,:]+bundle_yoffset_arcsec
    
    if return_params:
        params = {'bundle_name':bundle_name,
                  'fibers_per_side':fibers_per_side,
                  'n_observations':n_observations,
                  'rotation_degrees':rotation_degrees,
                  'bundle_xoffset_arcsec':bundle_xoffset_arcsec,
                  'bundle_yoffset_arcsec':bundle_yoffset_arcsec,
                  'fiber_diameter_arcsec':fiber_diameter_arcsec,
                  'core_diameter_arcsec':core_diameter_arcsec,
                  'cladding_arcsec':cladding_arcsec,
                  'fibers_in_bundle':len(xc_arr)}
        return (xc_arr,yc_arr),params
    else: 
        return (xc_arr,yc_arr)
    
    
def Fiber_Observe(cube_data,core_x_pixels,core_y_pixels,core_diameter_pixels,return_weights=False):
    '''
    Produces an ndarray of losvds/spectra for each fiber applied to the data. 
    
    `cube_data` [numpy.ndarray] must be in format (`Nels`,`spatial_yels`,
    `spatial_xels`). `Nels` denotes the number of wavelength/velocity elements.
    `spatial_xels` and `spatial_yels` denote the spatial dimensions of the data.
    Consequently, `cube_data[0]` should return a slice of the cube with dimensions: 
    
    `cube_data[0].shape`: (`spatial_yels`,`spatial_xels`). 
    
    `core_x[y]_pixels`: [float,int,list,numpy.ndarray] 
        The `x`[`y`] (or column[row]) positions of the fiber core centroids. Can 
        be a single value (e.g. float) or an array/list of values for multiple 
        fibers. Must have a number of elements which matches `core_y_pixels`. 
        Used to determine the number of fibers to be applied. Values should be 
        in pixels (not, for example, arcsec).
    
    `core_diameter_pixels`: [float,int,list,numpy.ndarray]
        The diameter of each fiber core in pixels. The number of elements must 
        either match `core_x[y]_pixels` OR be a single value for all fibers. 
        In the latter scenario, it is assumed that all cores have the same
        diameter.
    
    Returns:
    
        ndarray with shape (`N_fibers`, `Nels`) where each row 
    is the spectra/losvd "observed" by the fiber in the data. The algorithm first 
    selects a rectangular set of pixels around the fiber in the data. These pixels
    are then further refined spatially by a factor which guarantees that there are 
    at least 100 spatial elements along the diameter of the fiber. The number of 
    sub-pixels within each proper pixel within the fiber is then computed to 
    estimate the area of each pixel subtended by the fiber. The resulting weight 
    map is applied to the data at each spectral/losvd slice to produce a single 
    fiber array.
    
    if return_weights (default False):
    
        ndarray with shape (N_fibers,spatial_y,spatial_x) which contains weight maps
    for the contribution of each fiber to each pixel in the input grid.
    '''
    
    data_shape = cube_data.shape
    if len(data_shape) != 3:
        raise Exception("Data must have three axes with dimensions (Nels,spatial_y,spatial_x). Stopping...")
    size_y, size_x, Nels = data_shape[1],data_shape[2],data_shape[0]
    
    if type(core_x_pixels) in [float,int]: 
        core_x_pixels = np.array([core_x_pixels,]).astype(float)
    elif type(core_x_pixels) in [list,np.ndarray]:
        core_x_pixels = np.array(core_x_pixels).astype(float)
    else:
        try: 
            core_x_pixels = np.array([float(core_x_pixels),])
        except:
            raise Exception("core_x_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_y_pixels) in [float,int]:
        core_y_pixels = np.array([core_y_pixels,]).astype(float)
    elif type(core_y_pixels) in [list,np.ndarray]:
        core_y_pixels = np.array(core_y_pixels).astype(float)
    else:
        try: 
            core_y_pixels = np.array([float(core_y_pixels),])
        except:
            raise Exception("core_y_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_diameter_pixels) in [float,int]:
        core_diameter_pixels = np.array([core_diameter_pixels,]).astype(float)
    elif type(core_diameter_pixels) in [list,np.ndarray]:
        core_diameter_pixels = np.array(core_diameter_pixels).astype(float)
    else:
        try: 
            core_diameter_pixels = np.array([float(core_diameter_pixels),])
        except:
            raise Exception("core_diameter_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    # check that x,y core position array dimensions match
    if core_x_pixels.shape != core_y_pixels.shape:
        raise Exception("Fiber core x- and y- position arrays (or lists/values) do not have matching dimensions. Stopping...")
    
    N_fibers = core_x_pixels.shape[0]
    # core radius not necessarily constant but may be particular to each fiber
    if core_diameter_pixels.shape[0] not in [1,N_fibers]:
        raise Exception("Fiber core_diameter_pixels must either be a single float (all/any fibers have the same diameter) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")
    core_radius_pixels = core_diameter_pixels/2
    core_radius_pixels = core_radius_pixels.reshape(-1,1,1)
    core_x_pixels = core_x_pixels.reshape(-1,1,1)
    core_y_pixels = core_y_pixels.reshape(-1,1,1)
    
    Y,X = np.ogrid[0:size_y,0:size_x]
    Y = Y[np.newaxis,...]
    X = X[np.newaxis,...]
    
    # initialize weight map
    weight_map = np.zeros((N_fibers,size_y,size_x)).astype(int)

    # select rectangular region around fiber to refine for weight estimates
    weight_map[(np.abs(X+0.5-core_x_pixels)<core_radius_pixels+0.5) * 
               (np.abs(Y+0.5-core_y_pixels)<core_radius_pixels+0.5)] = 1
    
    indices = np.argwhere(weight_map)
    slices,rows,cols = indices[:,0],indices[:,1],indices[:,2]
    
    row_min = [np.min(rows[slices==i]) for i in range(N_fibers)]
    row_max = [np.max(rows[slices==i])+1 for i in range(N_fibers)]
    col_min = [np.min(cols[slices==i]) for i in range(N_fibers)]
    col_max = [np.max(cols[slices==i])+1 for i in range(N_fibers)]
    
    # the refined grid is defined to have a minimum of 100 pixels
    # across the diameter of the fiber
    rfactor = (100/core_diameter_pixels).astype(int)
    # if the diameter exceeds 100 pixels already, 
    # use the original regular grid
    rfactor[rfactor<1]=1
    
    # handling condition where a single core diameter is given
    if len(rfactor)==1 and N_fibers>1:
        core_radius_pixels = np.ones((N_fibers),dtype=int).flatten()*core_radius_pixels.flatten()
        rfactor = np.ones((N_fibers),dtype=int)*rfactor.flatten()
        
    weight_map = weight_map.astype(float)
    core_array = np.zeros((N_fibers,Nels)) 

    # Each refined grid is unique to the scale factor.
    # Each original grid can also differ. Therefore, need loop.
    for i in np.arange(N_fibers):
        
        col_max_ = col_max[i]
        col_min_ = col_min[i]
        row_max_ = row_max[i]
        row_min_ = row_min[i]
        rfactor_ = rfactor[i]
        core_x_pixels_ = core_x_pixels[i]
        core_y_pixels_ = core_y_pixels[i]
        core_radius_pixels_ = core_radius_pixels[i]
        
        rsize_x,rsize_y = (col_max_-col_min_)*rfactor_,(row_max_-row_min_)*rfactor_
        Yr,Xr = np.ogrid[0:rsize_y,0:rsize_x]
        maskr = np.zeros((rsize_y,rsize_x))
        
        maskr[np.sqrt((Xr+col_min_*rfactor_+0.5 - core_x_pixels_*rfactor_)**2 +
                      (Yr+row_min_*rfactor_+0.5 - core_y_pixels_*rfactor_)**2) < core_radius_pixels_*rfactor_ ]=1
 
        weight_mapr = np.zeros((size_y*rfactor_,size_x*rfactor_))
        weight_mapr[row_min_*rfactor_:row_max_*rfactor_,col_min_*rfactor_:col_max_*rfactor_] = maskr
        patch = maskr.reshape(row_max_-row_min_,rfactor_,col_max_-col_min_,rfactor_).sum(axis=(1,3)).astype(float)/rfactor_**2
        weight_map[i,row_min_:row_max_,col_min_:col_max_]=patch
        core_array[i] = np.sum(cube_data*weight_map[i].reshape(1,size_y,size_x),axis=(1,2))
        
#     cube_data = cube_data[np.newaxis,...] 
#     core_array = np.sum(cube_data*weight_map.reshape(N_fibers,1,size_y,size_x),axis=(2,3))

    return (core_array,weight_map) if return_weights else core_array


def Change_Coords(core_x_pixels,core_y_pixels,core_diameter_pixels,
                  input_grid_dims,output_grid_dims):
    
    '''
    This tool is used to map the fiber core centroid locations on the image plane to a new
    grid covering the same field of view (FOV) but with an arbitrary pixel scale. This pixel
    scale is set by the ratio of `input_grid_dims` to `output_grid_dims`. Consequently, the 
    coordinates of objects in an input grid with dimensions (10,10) could be mapped to (3,3)
    with a scale of 0.333.
    
    Output FOV is always the same as the input FOV! The scale is the only thing that differs.
    The scale must be equal in both x- and y- dimensions. Consequently, an initial grid of 
    (10,15) cannot be scaled to (5,3) as this would require a scale of 2 in the y-direction 
    and a scale of 5 in the x-direction. 
    
    The output `core_x_pixels` and `core_y_pixels` will have a coordinate system defined with
    with (0,0) at the upper left corner of the image. Therefore, with output grid dimensions 
    of (10,10), the center of the fiber array would be (5,5).
    
    Argument descriptions are the same as for the Fiber_Observe function.
    
    `input_grid_dims` [int,tuple]: 
    The dimensions of the input grid in which the coordinates of the fiber cores are set.
    
     `output_grid_dims` [int,tuple]: 
    The dimensions of the output grid into which the coordinates of the fiber cores are to 
    be determined.
    
    Returns:
    
    `core_x_pixels`,`core_y_pixels`,`core_diameter_pixels` scaled to the new grid dimensions.
    
    
    '''
    
    if type(input_grid_dims)==tuple and type(output_grid_dims)==tuple:
        try:
            osize_y,osize_x = output_grid_dims[0],output_grid_dims[1]
            size_y,size_x = input_grid_dims[0],input_grid_dims[1]
        except:
            raise Exception('Grid dimensions are tuples but do not contain two elements. Stopping...')
    elif type(input_grid_dims)==int and type(output_grid_dims)==int:
        osize_y,osize_x = output_grid_dims,output_grid_dims
        size_y,size_x = input_grid_dims,input_grid_dims
    else:
        raise Exception('Grid dimensions must either be both tuples, e.g. (nrows,ncols), or both ints. Stopping...')
        
    if type(core_x_pixels) in [float,int]: 
        core_x_pixels = np.array([core_x_pixels,]).astype(float)
    elif type(core_x_pixels) in [list,np.ndarray]:
        core_x_pixels = np.array(core_x_pixels).astype(float)
    else:
        try: 
            core_x_pixels = np.array([float(core_x_pixels),])
        except:
            raise Exception("core_x_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_y_pixels) in [float,int]:
        core_y_pixels = np.array([core_y_pixels,]).astype(float)
    elif type(core_y_pixels) in [list,np.ndarray]:
        core_y_pixels = np.array(core_y_pixels).astype(float)
    else:
        try: 
            core_y_pixels = np.array([float(core_y_pixels),])
        except:
            raise Exception("core_y_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_diameter_pixels) in [float,int]:
        core_diameter_pixels = np.array([core_diameter_pixels,]).astype(float)
    elif type(core_diameter_pixels) in [list,np.ndarray]:
        core_diameter_pixels = np.array(core_diameter_pixels).astype(float)
    else:
        try: 
            core_diameter_pixels = np.array([float(core_diameter_pixels),])
        except:
            raise Exception("core_diameter_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    # check that x,y core position array dimensions match
    if core_x_pixels.shape != core_y_pixels.shape:
        raise Exception("Fiber core x- and y- position arrays (or lists/values) do not have matching dimensions. Stopping...")
    
    N_fibers = core_x_pixels.shape[0]
    # core radius not necessarily constant but may be particular to each fiber
    if core_diameter_pixels.shape[0] not in [1,N_fibers]:
        raise Exception("Fiber core_diameter_pixels must either be a single float (all/any fibers have the same diameter) or an array/list of length equal to `core_x_pixels` and `core_y_pixels`. Stopping...")
    
    scale = float(osize_y)/size_y
    scale_x = float(osize_x)/size_x
    if scale_x != scale:
        raise Exception("The scale by which the input coordinates are converted to output coordinates must be the same in x- and y- dimensions. Make sure that `input_grid_dims` and `output_grid_dims` satisfy this condition. Stopping...")
    
    core_x_pixels *= scale
    core_y_pixels *= scale
    core_diameter_pixels *= scale
    
    return core_x_pixels.flatten(),core_y_pixels.flatten(),core_diameter_pixels.flatten()
    
    
def Fiber_to_Grid(fiber_data,core_x_pixels,core_y_pixels,core_diameter_pixels,grid_dimensions_pixels,
                  use_gaussian_weights=False,gaussian_sigma_pixels=1.4,rlim_pixels=None,
                  use_broadcasting=False):
    
    '''
    With a fiber core array [ndarray] with shape (`N_fibers`,`Nels`) along with their
    x- and y- coordinates (centroid) on a grid of dimensions `grid_dimensions_pixels`,
    this tool computes the intensity contribution of each fiber to each pixel of the grid.
    There are two options:
    
    (1) The intensity is distributed uniformly over all pixels completely contained within
    the fiber and partially within pixels that partially overlap with the fiber. In
    this respect, the method is exactly analogous to Fiber_Observe but in reverse. 
    Pixels that partially overlap with the fiber receive a portion of the intensity that
    is weighted by fraction of area of overlap with respect to a full pixel size. This
    method conserves intensity. This is checked by taking the sum along axis=0 of 
    `fiber_data` (the fiber axis) and comparing to the sum in each wavelength/losvd 
    slice (axis=(1,2)) of the output datacube. The output cube will therefore have a
    total sum that is equal to the sum of fiber_data, but distributed on the grid. 
    
    (2) The weights are determined by adopting a gaussian distribution of the intensity from
    each fiber on the output grid. This method emulates the SDSS-IV MaNGA data reduction
    pipeline. Specifically, see LAW et al. 2016 (AJ,152,83), Section 9.1 on the 
    construction of the regularly sampled cubes from the non-regularly sampled fiber
    data. The intensity contribution from each fiber, f[i], to each regularly spaced output
    pixel is mapped by a gaussian distribution:
    
    w[i,j] = exp( -0.5 * r[i,j]^2 / sigma^2 )
    
    where r[i,j] is the distance between the fiber core and the pixel centroid and 
    sigma is a constant of decay (taken to by 0.7 arcsec for MaNGA, for example). 
    These weights are necessarily normalized to conserve intensity. 
    
    W[i,j] = w[i,j] / SUM(w[i,j]) from k=1 to N_fibers 
    
    Where the sum is over all fibers. Note that there is a distinction between the 
    N_fibers used in LAW et al. 2016 and the one used here. Here, N_fibers refers
    to all fibers from all exposures (equivalent to the N used by LAW et al. 2016). 
    Additionally, the `alpha` parameter used in LAW et al. 2016 is computed and applied
    to the intensities. `alpha` converts the "per fiber" intensity to the "per pixel"
    intensity in the new grid. The resulting `out_cube` conserves intensity from the 
    original cube in the limit where there is adequate sampling of the original cube
    by the fibers. With sparse sampling, the intensity is not necessarily conserved.
    Note that this differes from (1) which only conserves intensities within the 
    fibers themselves. Method (2) also allows a scale, `rlim_pixels` beyond which a 
    fiber contributes no intensity in the output grid (weights are zero). 
    
    `fiber_data` [np.ndarray] with shape (N_fibers,Nels):
    Fiber data arrays. These contain the spectra measured in each fiber to be 
    distributed on the output grid.
    
    `grid_dimensions_pixels`: [int,tuple]
    Dimensions of the output grid. If tuple, should be the (spatial_x,spatial_y)
    shape of the output grid.
    
    `use_gaussian_weights` [boolean]:
    If False (default), use method (1) outlined above. If True, use method (2). 
    If True, the `gaussian_sigma_pixels` and `rlim_pixels` are used to determine 
    the profile of the gaussian distribution used in the weights.
    
    `gaussian_sigma_pixels` [int,float,list,np.ndarray]:
    The characteristic size of the 2d circular gaussian used when
    `use_gaussian_weights` is True. 
    
    `rlim_pixels` [int,float,list,np.ndarray]:
    The distance in pixels from a fiber core beyond which the weights assigned 
    to all pixels in a weight map are zero (default None, i.e. the weights extend
    to infinity). 
    
    `use_broadcasting` [boolean]:
    The broadcasting of the weight maps with the spectra from each fiber to produce
    the output datacubes can be very memory-intensive. You can estimate the memory
    demand by computing (N_fibers*Nels*output_spatial_y*output_spatial_x*64/8/1e9)
    for the size of the object that needs to be summed over the N_fibers dimension
    in Gigabytes. If this exceeds your memory requirements, `use_broadcasting` should
    be set to False. This will greatly increase the computation time at the expense
    of memory intensiveness.
    
    '''
    
    fiber_data = np.array(fiber_data,dtype=float)
    data_shape = fiber_data.shape
    if len(data_shape) == 1:
        N_fibers,Nels = 1,data_shape[0]
        fiber_data = fiber_data.reshape(1,Nels)
    elif len(data_shape) == 2:
        N_fibers,Nels = data_shape[0],data_shape[1]
    else:
        raise Exception("fiber_data can have either one or two axes. No more, no less. Stopping...")
    
    if type(core_x_pixels) in [float,int]: 
        core_x_pixels = np.array([core_x_pixels,]).astype(float)
    elif type(core_x_pixels) in [list,np.ndarray]:
        core_x_pixels = np.array(core_x_pixels).astype(float)
    else:
        try: 
            core_x_pixels = np.array([float(core_x_pixels),])
        except:
            raise Exception("core_x_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_y_pixels) in [float,int]:
        core_y_pixels = np.array([core_y_pixels,]).astype(float)
    elif type(core_y_pixels) in [list,np.ndarray]:
        core_y_pixels = np.array(core_y_pixels).astype(float)
    else:
        try: 
            core_y_pixels = np.array([float(core_y_pixels),])
        except:
            raise Exception("core_y_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    if type(core_diameter_pixels) in [float,int]:
        core_diameter_pixels = np.array([core_diameter_pixels,]).astype(float)
    elif type(core_diameter_pixels) in [list,np.ndarray]:
        core_diameter_pixels = np.array(core_diameter_pixels).astype(float)
    else:
        try: 
            core_diameter_pixels = np.array([float(core_diameter_pixels),])
        except:
            raise Exception("core_diameter_pixels not in accepted format. Use a list, numpy array, int, or float. Stopping...")
        
    # check that x,y core position array dimensions match
    if core_x_pixels.shape != core_y_pixels.shape:
        raise Exception("Fiber core x- and y- position arrays (or lists/values) do not have matching dimensions. Stopping...")
    
    # core radius not necessarily constant but may be particular to each fiber
    if core_diameter_pixels.shape[0] not in [1,N_fibers]:
        raise Exception("Fiber core_diameter_pixels must either be a single float (all/any fibers have the same diameter) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")
    core_radius_pixels = core_diameter_pixels/2
    core_radius_pixels = core_radius_pixels.reshape(-1,1,1)
    core_x_pixels = core_x_pixels.reshape(-1,1,1)
    core_y_pixels = core_y_pixels.reshape(-1,1,1)

    if type(grid_dimensions_pixels) == int:
        size_y,size_x = grid_dimensions_pixels,grid_dimensions_pixels
    elif type(grid_dimensions_pixels) == tuple:
        size_y,size_x = int(grid_dimensions_pixels[1]),int(grid_dimensions_pixels[0])
    else:
        raise Exception("grid_dimensions_pixels must be an int or a tuple of two ints (e.g. '(10,10)')")
        
    Y,X = np.ogrid[0:size_y,0:size_x]
    Y = Y[np.newaxis,...]
    X = X[np.newaxis,...]
    
    if not use_gaussian_weights:
        
        # initialize weight map
        weight_map = np.zeros((N_fibers,size_y,size_x)).astype(int)

        # select rectangular region around fiber to refine for weight estimates
        weight_map[(np.abs(X+0.5-core_x_pixels)<core_radius_pixels+0.5) * 
                   (np.abs(Y+0.5-core_y_pixels)<core_radius_pixels+0.5)] = 1

        indices = np.argwhere(weight_map)
        slices,rows,cols = indices[:,0],indices[:,1],indices[:,2]

        row_min = [np.min(rows[slices==i]) for i in range(N_fibers)]
        row_max = [np.max(rows[slices==i])+1 for i in range(N_fibers)]
        col_min = [np.min(cols[slices==i]) for i in range(N_fibers)]
        col_max = [np.max(cols[slices==i])+1 for i in range(N_fibers)]

        # the refined grid is defined to have a minimum of 100 pixels
        # across the diameter of the fiber
        rfactor = (100/core_diameter_pixels).astype(int)
        # if the diameter exceeds 100 pixels already, 
        # use the original regular grid
        rfactor[rfactor<1]=1    

        # handling condition where a single core diameter is given
        if len(rfactor)==1 and N_fibers>1:
            core_radius_pixels = np.ones((N_fibers),dtype=int).flatten()*core_radius_pixels.flatten()
            rfactor = np.ones((N_fibers),dtype=int)*rfactor.flatten()
        weight_map = weight_map.astype(float)
        
        # Each refined patch is unique to the scale factor.
        # Each original patch can also differ. Therefore, need loop.
        for i in np.arange(N_fibers):

            col_max_ = col_max[i]
            col_min_ = col_min[i]
            row_max_ = row_max[i]
            row_min_ = row_min[i]
            rfactor_ = rfactor[i]
            core_x_pixels_ = core_x_pixels[i]
            core_y_pixels_ = core_y_pixels[i]
            core_radius_pixels_ = core_radius_pixels[i]

            rsize_x,rsize_y = (col_max_-col_min_)*rfactor_,(row_max_-row_min_)*rfactor_
            Yr,Xr = np.ogrid[0:rsize_y,0:rsize_x]
            maskr = np.zeros((rsize_y,rsize_x))

            maskr[np.sqrt((Xr+col_min_*rfactor_+0.5 - core_x_pixels_*rfactor_)**2 +
                          (Yr+row_min_*rfactor_+0.5 - core_y_pixels_*rfactor_)**2) < core_radius_pixels_*rfactor_ ]=1
            
            weight_mapr = np.zeros((size_y*rfactor_,size_x*rfactor_))
            weight_mapr[row_min_*rfactor_:row_max_*rfactor_,col_min_*rfactor_:col_max_*rfactor_] = maskr
            patch = maskr.reshape(row_max_-row_min_,rfactor_,col_max_-col_min_,rfactor_).sum(axis=(1,3)).astype(float)/rfactor_**2
            weight_map[i,row_min_:row_max_,col_min_:col_max_]=patch
            
        # mask weights where data is masked to avoid over-normalization
        weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)*np.ones((1,Nels,1,1))
        weight_map[np.isnan(fiber_data),:,:] = np.nan
        weight_map = weight_map.reshape(N_fibers,size_y,size_x)
        
        _weight_map_ = copy(weight_map)
        alpha = np.nansum(weight_map,axis=(1,2)).reshape(-1,1,1)
        normalization = np.nansum(weight_map,axis=0)
        normalization[normalization==0]=np.nan
        weight_map/=normalization
        weight_map[np.isnan(weight_map)]=0.
        weight_map/=alpha
            
        if not use_broadcasting:
            # code to reduce memory demand
            out_cube = np.zeros((Nels,size_y,size_x))
            fiber_data = fiber_data.reshape(N_fibers,Nels,1,1)
            weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)
            for i in range(N_fibers):
                out_fiber = fiber_data[i]*weight_map[i]
                out_fiber[np.isnan(out_fiber)]=0.
                out_cube += out_fiber
        else:
            out_cube = np.nansum(fiber_data.reshape(N_fibers,Nels,1,1)*weight_map.reshape(N_fibers,1,size_y,size_x),axis=0)
        
        return out_cube,_weight_map_
    
    else:
        
        # check `gaussian_sigma_pixels` type
        if type(gaussian_sigma_pixels) in [float,int]:
            gaussian_sigma_pixels = np.array([gaussian_sigma_pixels,]).astype(float)
        elif type(gaussian_sigma_pixels) in [list,np.ndarray]:
            gaussian_sigma_pixels = np.array(gaussian_sigma_pixels).astype(float)
        else:
            try: 
                gaussian_sigma_pixels = np.array([float(gaussian_sigma_pixels),])
            except:
                raise Exception("`gaussian_sigma_pixels` not in accepted format. Use a list, numpy array, int, or float. Stopping...")

        # gaussian_sigma_pixels not necessarily constant but may be particular to each fiber
        if gaussian_sigma_pixels.shape[0] not in [1,N_fibers]:
            raise Exception("`gaussian_sigma_pixels` must either be a single float (all/any fibers have the same sigma) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")

        # handling condition where a gaussian_sigma_pixels is given
        if len(gaussian_sigma_pixels)==1 and N_fibers>1:
            gaussian_sigma_pixels = np.ones((N_fibers),dtype=int).flatten()*gaussian_sigma_pixels.flatten()
        
        # generate cube of 2d gaussian pdfs with output grid dimensions
        r2 = ((X+0.5)-core_x_pixels)**2 + ((Y+0.5)-core_y_pixels)**2
        weight_map = np.exp(-0.5*r2/gaussian_sigma_pixels.reshape(N_fibers,1,1)**2)
        
        # for each fiber, all pixel weights outside `rlim_pixels` are zero
        if rlim_pixels is not None:
            
            if type(rlim_pixels) in [float,int]:
                rlim_pixels = np.array([rlim_pixels,]).astype(float)
            elif type(rlim_pixels) in [list,np.ndarray]:
                rlim_pixels = np.array(rlim_pixels).astype(float)
            else:
                try: 
                    rlim_pixels = np.array([float(rlim_pixels),])
                except:
                    raise Exception("`rlim_pixels` not in accepted format. Use a list, numpy array, int, or float. Stopping...")

            # gaussian_sigma_pixels not necessarily constant but may be particular to each fiber
            if rlim_pixels.shape[0] not in [1,N_fibers]:
                raise Exception("`rlim_pixels` must either be a single float (all/any fibers have the same sigma) or an array/list of length equal to core_x_pixels and core_y_pixels. Stopping...")

            # handling condition where a gaussian_sigma_pixels is given
            if len(rlim_pixels)==1 and N_fibers>1:
                rlim_pixels = np.ones((N_fibers),dtype=int).flatten()*rlim_pixels.flatten()
        
            weight_map[np.sqrt(r2)>rlim_pixels.reshape(N_fibers,1,1)] = 0
        
        # normalization to determine intensity contribution of each fiber to each pixel
        # as in LAW et al 2016
        
#         # mask weights where data is masked to avoid over-normalization
#         weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)*np.ones((1,Nels,1,1))
#         weight_map[np.isnan(fiber_data),:,:] = np.nan
#         weight_map = weight_map.reshape(N_fibers,size_y,size_x)
                                        
#         _weight_map_ = copy(weight_map)                   
#         alpha = 1./(np.pi*(core_radius_pixels)**2)
#         normalization = np.nansum(weight_map,axis=0)
#         normalization[normalization==0]=np.nan
#         weight_map/=normalization
#         weight_map*=alpha
#         weight_map[np.isnan(weight_map)]=0.
        
#         if not use_broadcasting:
#             # code to reduce memory demand
#             out_cube = np.zeros((Nels,size_y,size_x))
#             fiber_data = fiber_data.reshape(N_fibers,Nels,1,1)
#             weight_map = weight_map.reshape(N_fibers,1,size_y,size_x)
#             for i in range(N_fibers):
#                 out_fiber = fiber_data[i]*weight_map[i]
#                 out_fiber[np.isnan(out_fiber)]=0.
#                 out_cube += out_fiber
#         else:
#             out_cube = np.nansum(fiber_data.reshape(N_fibers,Nels,1,1)*weight_map.reshape(N_fibers,1,size_y,size_x),axis=0)
        
        alpha = 1./(np.pi*(core_radius_pixels)**2)
        
        if not use_broadcasting:
            out_cube = np.zeros((Nels,size_y,size_x))
            bar = FillingCirclesBar('Spatial reconstruction', max=Nels)
            for i in range(Nels):
                weight_map_el = copy(weight_map)
                weight_map_el[np.isnan(fiber_data[:,i])] = np.nan
                normalization_el = np.nansum(weight_map_el,axis=0)
                normalization_el[normalization_el==0]=np.nan
                weight_map_el/=normalization_el
                weight_map_el*=alpha
                weight_map_el[np.isnan(weight_map)]=0.
                out_el = np.nansum(fiber_data[:,i].reshape(N_fibers,1,1)*weight_map_el,axis=0)
                out_el[np.isnan(out_el)]=0.
                out_cube[i] = out_el
                bar.next()
            bar.finish()

        return out_cube,weight_map
    
    
if __name__ == '__main__':
    
    import numpy as np
    import os,sys,time
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='in',length=10,width=3,labelsize=20,right=1,top=1)
    ax.tick_params(axis='both',which='minor',direction='in',length=5,width=2,right=1,top=1)
    ax.tick_params(axis='both',which='major',pad=10)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    ax.set_xlabel(r'$x$-position [arcsec]',fontsize=20,labelpad=10)
    ax.set_ylabel(r'$y$-position [arcsec]',fontsize=20)
    ax.scatter(x=0.,y=0.,marker='+',s=100,zorder=10,color='black')
    
    (xc_arr,yc_arr),params = MaNGA_Observe(bundle_name='None',fibers_per_side=4,n_observations='Classic',
                                           bundle_xoffset_arcsec=0.,
                                           bundle_yoffset_arcsec=0.,
                                           rotation_degrees = 90.,
                                           return_params=True)
    fiber_diameter_arcsec = params['fiber_diameter_arcsec']
    core_diameter_arcsec = params['core_diameter_arcsec']
    n_observations = xc_arr.shape[-1]
    
    for i_obs in range(n_observations):
        xc_obs,yc_obs = xc_arr[:,i_obs],yc_arr[:,i_obs]
        for xy in zip(xc_obs,yc_obs):
            clad = Circle(xy=xy,radius=fiber_diameter_arcsec/2,transform=ax.transData,edgecolor='black',facecolor='White',alpha=0.3)
            ax.add_artist(clad)
            core = Circle(xy=xy,radius=core_diameter_arcsec/2,transform=ax.transData,edgecolor='None',facecolor='Grey',alpha=0.3)
            ax.add_artist(core)

    fig.savefig('MaNGA_Observe_Test_N37_rot90.pdf',bbox_inches='tight')
    
    sys.exit(0)