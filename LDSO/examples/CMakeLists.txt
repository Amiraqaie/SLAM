# we have three types of example binaries.
# All target diffrent types of dataset 
# namely TUM_MONO and EURCO and KITTI

add_executable(test test.cpp)
add_executable(run_ldso_tum_mono run_ldso_tum_mono.cc)

target_link_libraries(test
    ldso_ngd  
    ${THIRD_PARTY_LIBS})

target_link_libraries(run_ldso_tum_mono
    ldso_ngd  
    ${THIRD_PARTY_LIBS})

# here follows EUROC and KITTI
