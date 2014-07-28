import logging
import numpy as np
import matplotlib.pyplot as plt
import h5py
from default_parser import DefaultArgumentParser

logger=logging.getLogger(__file__)

def plot_single(traj_dset):
    cnt=len(traj_dset)
    logger.debug("{0} values in dataset".format(cnt))
    # The dataset is an array of tuples, not a 2d array.
    s=np.zeros(cnt, np.int64)
    i=np.zeros(cnt, np.int64)
    r=np.zeros(cnt, np.int64)
    t=np.zeros(cnt, np.double)
    for idx, val in enumerate(traj_dset):
        s[idx], i[idx], r[idx], t[idx]=val

    # Does t sometimes go backwards? By how much?
    tmax=t[0] # largest seen
    timax=0   # largest difference between now and max.
    for idx in range(1,len(t)):
        if t[idx]<tmax:
            timax=max(tmax-t[idx], timax)
        tmax=max(t[idx], tmax)
    print("A value was {0} less than a previous value".format(timax))

    fix, ax=plt.subplots()
    ax.plot(t, s, 'k', label='S')
    ax.plot(t, i, 'r', label='I')
    ax.plot(t, r, 'g', label='R')
    plt.xlabel('time [years]')
    plt.ylabel('individuals')
    legend=ax.legend(loc='upper center', shadow=False)
    plt.show()



def foreach_trajectory(f, func):
    trajectories=f['/trajectory']

    for trajectory_name in trajectories:
        dset=trajectories[trajectory_name]
        func(dset)


def showds(ds):
    print(ds)
    attributes=list()
    for x in ds.attrs:
        attributes.append("{0} {1}".format(x, ds.attrs[x][0]))
    print("  {0}".format(", ".join(attributes)))


def showproginfo(f):
    attrs=f['/trajectory'].attrs
    for x in attrs:
        if (attrs[x].dtype.type is np.bytes_):
            print("{0}: ".format(x))
            for v in attrs[x]:
                print(v.decode())
        else:
            print("{0}: {1}".format(x, attrs[x]))


def eeid_long_behavior():
    B=1/70
    beta=400
    mu=1/70
    gamma=365/14
    S=(mu+gamma)*B/(beta*mu)
    I=(beta-mu-gamma)*B/(beta*(mu+gamma))
    R=gamma*I/mu
    print("EEID long time is\n\tS\t{0}\n\tI\t{1}\n\tR\t{2}".format(S, I, R))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser=DefaultArgumentParser(description="Quick look at an H5 file")
    parser.add_function("info", "Find what program made the file.")
    parser.add_function("trajectory", "Plot the trajectory")
    parser.add_function("dir", "List datasets")
    parser.add_function("eeid", "verify eeid example values")
    parser.add_argument("--file", dest="file", action="store",
        default="sirexp.h5", help="data file to read")

    args=parser.parse_args()

    filename=args.file
    f=h5py.File(filename, "r")

    if args.info:
        showproginfo(f)
    if args.trajectory:
        foreach_trajectory(f, plot_single)
    if args.dir:
        foreach_trajectory(f, showds)
    if args.eeid:
        eeid_long_behavior()
    #foreach_trajectory("sirexp.h5", plot_single)
    if not parser.any_function():
        parser.print_help()

