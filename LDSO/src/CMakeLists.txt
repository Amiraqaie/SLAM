add_library(ldso_ngd
    # ALL of the source files
    Feature.cc
    Setting.cc
)

target_link_libraries(
    ldso_ngd
    ${THIRD_PARTY_LIBS}
)