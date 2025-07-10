import streamlit as st

testList = [0,1,3,1,1,11,1,1,1,1,1,18,1,11]


max_idx = testList.index(max(testList))

u = testList[max_idx]

print(max_idx, u)
