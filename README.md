# OpticalFlowToolkit
Set of tools for reading, visualizing,writting and computing error  optical flow in middlebury/kitti format.
Numpy warper to create OF class fully compatible. 
`pip3 install pynpflow`  
## Contents  
There are two main files. flowlib provides simple functions directly relying on the mentioned libraries. All of them takes flows/paths_to_flows as input, therefore they can be easily used for a certain task.  

flowlibwarped provides a subclass of numpy arrays which incorporates the functions defined in flowlib as methods of the array. It does not overwrite numpy functions/methods but complement them.

## Dependencies  
Mandatory:  
- numpy  

Optionals (For reading,writing or plotting some certain formats)  
 - pypng  
 - matplotlib  
 - imageio  
 - cv2  

# flowlibwarper  
This library allows to easily work with single/multiple optical flows.  

Example:  
```
from flowlibwarped import npflow as fl
flow = fl(input)
```
Input can be:
- a ```np.ndarray``` with ```shape = (N,H,W,C)``` or ```shape = (H,W,C)``` 
     N: Number of OF 
    C: OF channels  
- a path_to_of_file ```.flo/.png/.jpg/.jpeg```
- a list of ordered paths 
## Methods  and features  
```fl.fp2int(ui=8)``` returns a mapped conversion from floating point to np.uintN  
```fl.int2fp(ui=8)``` returns a mapped conversion from np.uintN to float  
```fl.asimage(display = False)```  returns a ```np.uin8``` middlebury color code image. Display optionally prints in console some data.  
```fl.scale(new_range,[dtype])``` returns the optical flow mapped into the desired range. new_range is a tuple or list with the new desired range and dtype a numpy dtype  
```write(filename,idx=None,workers=mp.cpu_count())```  
Stores optical flow in .flo/.png/.jpg files.  
    Inputs:  
    filename: str or list of strings. Path to save optical flow(s).  
    idx: int or list of int. In case of working with several optical flows, indices of the desired optical flow(s). Allows to save a single OF (set it as int) or several optical flows (set it as a list of indices which points to the OFs you want to save)  
    workers: integer. To speed up this process multiprocessing is applied. Set the number of threads to be used. Default: all

## Class iterator  
You can iterate over the optical flows in a simple way: 
```
from flowlibwarped import npflow as fl
flow = fl(input)

for flow_i in flow:
    do_your_tasks
```
But you can also iterate on a subset of the optical flows using numpy indexing:
```
for flow_i in flow[index0:indexN]:
    do_your_tasks
for flow_i in flow[::2]:
    do_your_tasks
```
## Calling the instance  
Calling the instance returns 3 different things depending on the synthax:
- No arguments: Returns the optical flow as a ```np.ndarray``` object.  
```
from flowlibwarped import npflow as fl
flow = fl(input)
flow() #Returns a np.ndarray of the optical flow
```
- integer argument:
```
flow(5) #Equivalent to flow[5]. No indexing allowed here.
```
- tuple argument: (idx_initial,idx_final,[step])
    Allows to set config the iterator. By default parameters goes form initial index 0 to the last OF and step 1.
    Note: The step parameters is copied when performing operations.
```

flow((5,10)) 
for flow_i in flow:
    ...
#Is equivalent to:
for flow_i in flow[5:10]:
    ...
    
#Note that if you modify the step value:
flow((5,10,2)) 
for flow_i in flow[::2]:
    ...
#You would be picking 1 out of 4 OFs there and after assigning a new subset
subset = flow[2:10]
subset.step #Will be equal to flow.step
```
# Optical Flow format supported  
At the time of reading or writting, the format conditions the io. To properly save an optical flow the format of the filename parsed to a function must be either.flo, .jpg or .png
```.jpg / .jpeg```:
Reads/writes a 3-channel uint8 jpg image being the channel 1 and 2 the components of the OF. The channel 3 is filled with zeros at the time of writting.
This is a very fast method since it relies on scipy libraries. The average weight of a file is ~15 kBytes for a 300x640 OF.  
```.png```
Reads a 3 channel .png image (Kitty format) or a 2-channel .png image of type uint16. Writes a 2-channel .png image type uint16 (uint8 allowed)  
```.flo```  
Read/writes the standar .flo format
