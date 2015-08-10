sr      =  44100
;kr      =  100
ksmps   =  16
nchnls  =  1
0dbfs = 4.0

/* Bartlett window */
itmp    ftgen 1, 0, 1024, 20, 3, 1
itmp    ftgen 2, 0, 16384, 7, 1, 16384, -1
/* sine */
itmp    ftgen 4, 0, 1024, 10, 1

        instr grain
        inow = p2
        idur = p3
        iamp = p4
        iframe = p5
        ifdur  = p6
        iphase = p7
        ipitch = 0.5
aenv    oscili iamp, 1/idur, 1
aa01    poscil iamp, ipitch, 101, iphase
        out aenv*(aa01)
        endin

