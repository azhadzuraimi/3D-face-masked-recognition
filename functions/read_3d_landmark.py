
from scanf import scanf

def read_lm3file(filepath):
    try:
        fid = open(filepath, 'r')
        print ("Name of the file: ", fid.name)
    except FileNotFoundError:
        print(f"File {fname} not found.  Aborting")
        sys.exit(1)
    except OSError:
        print(f"OS error occurred trying to open {fname}")
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error opening {fname} is",repr(err))
        sys.exit(1)  # or replace this with "raise" ? 
    else:
        with fid:
            fid.readline()
            if fid.readline()=='*':
            #here after in the file, some info text is written -> skip them
                while (fid.readline()!=''):
                    pass
            N = scanf('%d',fid)
            labels = cell(N,1)
            pts3D = zeros(3,N)
            fid.readline()
            for n in range(1, N):
                labels[n] = fid.readline()
                pts3D[:,n] = scanf(fid, '%g', 3)
                fid.readline()
            fclose(fid)
    return pts3D, labels
