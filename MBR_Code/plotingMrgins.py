import numpy as np
margins = []
beta = 0.1
dataset = 'strategyqaT'
bm_margins = []
mw_margins = []
for i in range (4):
            rbm = np.load('../model-based-mbr/arraysBox/RBM{}_{}.npy'.format(i,dataset)).tolist()
            rmw = np.load('../model-based-mbr/arraysBox/RMW{}_{}.npy'.format(i,dataset)).tolist()
            if len(margins)==0:
                margins.append(0)
                mw_margins.append(0)
                bm_margins.append(0)

            rewardMarginsBestMiddle = [float(beta)*y for  y in rbm]
            rewardMarginsMiddleWorst = [float(beta)*y for  y in rmw]
            rewardMargins = rewardMarginsBestMiddle+ rewardMarginsMiddleWorst
            
            margins.append(sum(rewardMargins)/len(rewardMargins))
            bm_margins.append(sum(rewardMarginsBestMiddle)/len(rewardMarginsBestMiddle))
            mw_margins.append(sum(rewardMarginsMiddleWorst)/len(rewardMarginsMiddleWorst))

print('mw_margins ',mw_margins)
print('bm_margins ',bm_margins)
print('margins ',margins)
