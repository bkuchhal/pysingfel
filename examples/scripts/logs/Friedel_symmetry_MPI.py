import numpy as np
import time
import psana
from mpi4py import MPI
import argparse


def average_error_per_pixel(myJobs, ipx, ipy, radius, run, times, det):
    print myJobs
    avg_error_arr = np.zeros_like(myJobs, dtype=np.float32)
    idx = 0
    
    mask_pso = None
    flip_mask_pso = None
    circle_mask = None
    for i in myJobs:
        evt = run.event(times[i])
        img = det.image(evt)
        print "evt: ", i, img.shape
        if mask_pso is None:
            mask_pso = mask_psocake(det, evt)
        
        if circle_mask is None:
            circle_mask = circular_mask(img, ipx, ipy, radius)
            
        if flip_mask_pso is None:
            flip_mask_pso = flip_img(mask_pso)
        
        if img is not None:
            masked_img = img * circle_mask * mask_pso * flip_mask_pso
            crop = cropped_img(masked_img, ipx, ipy)
            no_pixels = np.count_nonzero(crop)
            error = flip_img(crop) - crop
            avg_error_arr[idx] = sum(sum(np.abs(error)))/no_pixels
        else:
            avg_error_arr[idx] = -1
            
        idx += 1
            
    return avg_error_arr


def getMyUnfairShare(numJobs,numWorkers,rank):
    """Returns number of events assigned to the slave calling this function."""
    if numJobs >= numWorkers:
        allJobs = np.arange(numJobs)
        jobChunks = np.array_split(allJobs,numWorkers)
        myChunk = jobChunks[rank]
        myJobs = allJobs[myChunk[0]:myChunk[-1]+1]
    else:
        if rank == 0:
            myJobs = np.arange(numJobs)
        else:
            myJobs = []
    return myJobs


def circular_mask(img, ipx, ipy, radius):
    X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    R = np.sqrt((X - ipx)**2 + (Y - ipy)**2)
    circle_mask = np.zeros_like(img, dtype=int)
    circle_mask[np.where(R < radius)] = 1
    return circle_mask


def mask_psocake(det, evt):
    mask = np.load('/reg/d/psdm/cxi/cxitut13/scratch/bhar544/amox26916/bhar544/psocake/r0190/mask.npy')
    mask = det.image(evt, mask)
    return mask


def cropped_img(img, ipx, ipy):
    
    # Maximum rectangle for image
    x_width = img.shape[0]
    y_width = img.shape[1]

    half_length_x = min([ipx, x_width - ipx - 1])
    half_length_y = min([ipy, y_width - ipy - 1])

    # Ewald slices
    img = img[(ipx-half_length_x+1):(ipx+half_length_x+1), (ipy-half_length_y+1):(ipy+half_length_y+1)]
    
    return img


def flip_img(img):
    return np.flipud(np.fliplr(img))


def main():
    
    # Parse user input
    params = parse_input_arguments(sys.argv)
    radius = params['radius']
    exp = params['exp']
    run = params['run']
    detname = params['det']
 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    numWorkers = comm.size
    print rank
    
    dsname = 'exp='+exp+':run='+str(run)+':idx'
    ds = psana.DataSource(dsname)
    run = ds.runs().next()
    times = run.times()
    env = ds.env()
    numEvents = len(times)
    evt = run.event(times[0])

    # setup detector
    det = psana.Detector(detname, env)
    ipx, ipy = det.point_indexes(evt, pxy_um=(0,0))
    
    myJobs = getMyUnfairShare(numEvents, numWorkers, rank)
    avg_error_arr = average_error_per_pixel(myJobs, ipx, ipy, radius, run, times, det)
    
    if rank == 0:
        
        start_time = time.time()        
        for i in range(1, numWorkers):
            avg_error_arr = np.append(avg_error_arr, comm.recv(source=i))
        
        np.save('Errors_amo86615_pnccdBack.npy', avg_error_arr)
        print 'Total time is: ' + str(time.time() - start_time) + 'seconds.' 

    else:
        comm.send(avg_error_arr, dest=0)
            

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--radius', type=np.float32, help='Maximum resolution required')
    parse.add_argument('-e', '--exp', type=str, help='Experiment Name')
    parse.add_argument('-r', '--run', type=int, help='Run number')
    parse.add_argument('-d', '--det', type=str, help='Detector')
    # convert argparse to dict
    return vars(parse.parse_args(args))


if __name__ == '__main__':
    main()

