"""Measure a blue outlet cup's curved white lower rim in current image pixels.

Station registration supplies only a bounded search neighborhood. No reference
cup pixels or previous-frame coordinates are returned as current observations.
"""
import cv2
import numpy as np

def detect(image,expected,width):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # A brighter exposure can bring blue-tinted black panels above the nominal
    # value cutoff. Test both calibrated luminance and exposure-scaled cutoffs;
    # each candidate still passes all shape, stripe and location checks.
    thresholds=sorted({50.,max(50.,.22*float(np.percentile(hsv[:,:,2],95)))})
    merged=[]
    for threshold in thresholds:
        for candidate in _detect_threshold(hsv,expected,width,threshold):
            candidate['blue_min_value']=threshold
            match=next((i for i,c in enumerate(merged) if np.linalg.norm(np.array(c['anchor'])-candidate['anchor'])<8),None)
            if match is None:merged.append(candidate)
            elif candidate['quality']>merged[match]['quality']:merged[match]=candidate
    return merged

def _detect_threshold(hsv,expected,width,threshold):
    # Exclude blue-tinted black machine panels. Extremely dark views may fail
    # closed; do not manufacture a cup from dark-background hue noise.
    blue=((hsv[:,:,0]>=90)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>=55)&(hsv[:,:,2]>=threshold)).astype(np.uint8)
    ex,ey=expected
    left=max(0,int(ex-width*.9-24));right=min(640,int(ex+width*.9+24))
    top=max(0,int(ey-45));bottom=min(480,int(ey+30))
    roi=blue[top:bottom,left:right]
    if roi.size==0:return []
    closed=cv2.morphologyEx(roi,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(closed,8)
    candidates=[]
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        if not .55*width<=w<=1.5*width or not 5<=h<=40 or area<80:continue
        mask=labels==label
        xs=[];ys=[]
        # Use central 80% of the blue band's lower boundary; side walls are
        # intentionally not treated as points on the bottom rim.
        for column in range(x+max(2,int(w*.1)),x+w-max(2,int(w*.1))):
            values=np.flatnonzero(mask[:,column])
            if len(values)<3:continue
            xs.append(column+left);ys.append(values[-1]+top)
        if len(xs)<max(16,int(w*.65)):continue
        xs=np.asarray(xs,float);ys=np.asarray(ys,float)
        midpoint=(xs[0]+xs[-1])/2;halfspan=(xs[-1]-xs[0])/2
        u=(xs-midpoint)/halfspan
        coefficients=np.polyfit(u,ys,2);fit=np.polyval(coefficients,u)
        rmse=float(np.sqrt(np.mean((fit-ys)**2)))
        sag=float(-coefficients[0])
        # A sloped straight edge is still flat after the linear term is removed.
        if not 1.3<=sag<=12 or rmse>1.5:continue
        center_y=float(np.polyval(coefficients,0))
        anchor=np.array([midpoint,center_y+2.])
        if np.linalg.norm(anchor-np.asarray(expected))>32:continue
        xi=xs.astype(int);yi=np.rint(ys).astype(int)
        white=[]
        for dx,dy in zip(xi,yi):
            band=hsv[min(479,dy+1):min(480,dy+6),dx]
            white.append(bool(len(band) and np.any((band[:,1]<80)&(band[:,2]>110))))
        white_fraction=float(np.mean(white))
        if white_fraction<.75:continue
        # Verify a dark-blue band above the rim across its width, not an
        # isolated curve in the stationary holder or a few printed characters.
        coverage=float(np.mean([np.mean(blue[max(0,int(py)-5):int(py)+1,int(px)]) for px,py in zip(xs,ys)]))
        if coverage<.65:continue
        candidates.append({'anchor':anchor.tolist(),
                           'box':[int(x+left-1),int(max(y+top-1,center_y-.32*w)),int(x+left+w+1),int(np.ceil(ys.max()+6))],
                           'white_rim_support':white_fraction,'blue_band_support':coverage,
                           'curve_sag_px':sag,'curve_rmse_px':rmse,
                           'quality':min(white_fraction,coverage)})
    return candidates
