import numpy as np

def ips_between(start, end):

    start = start.split(".")
    int_start_arry = np.array(start, dtype=int)

    end = end.split(".")
    int_end_arry = np.array(end, dtype=int)
    

    address1 = convert_ip_to_int(int_start_arry)
    address2 = convert_ip_to_int(int_end_arry)

    diff = address2 - address1
    return diff

def convert_ip_to_int(address):
    
    address_int = (256**3*address[0]) + (256**2*address[1]) + (256*address[2]) + (address[3])
    
    return address_int

    