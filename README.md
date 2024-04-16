# Stereo-Matching-Algorithms

This project is regarding stereo matching algorithms. I have opted 3 ways to do stereo matching. 

1. Guided Image Filter
2. Cost Aggregation using Mininmum Spanning tree and then stereo matching it
3. Non Local Cost Aggregation using Segment tree and then stereo matching it

There are also other ways to do stereo matching like the CensusTransform function, Belief propagation on trees, etc but we haven't opted those methods here

After the stero matching process we get output as a disparity map.
{Disparity map: A disparity map is a visual representation of the depth information extracted from a stereo image pair.}

Post processing works like guassian blur, normalisation have been done for the disparity maps.

We have used the middleburry dataset here for evaluation and experimental purposes 
