list(APPEND MATCHER_SOURCEFILES 
    matcher.cpp
    ../include/matcher.h
    base_matcher/base_matcher.h
    base_matcher/base_matcher.cpp
    Pattern_Matching/PatternMatching.h
    Pattern_Matching/PatternMatching.cpp

)

list(APPEND MATCHER_C_TYPE_SOURCEFILES 
    matcher_ctype.cpp
    ../include/matcher.h

)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(VERSIONINFO_RC "${CMAKE_BINARY_DIR}/VersionInfo.rc")
    configure_file("${CMAKE_SOURCE_DIR}/VersionInfo.rc.in"
                   "${VERSIONINFO_RC}")
    if (BUILD_SHARED_LIBS)
        add_library(templatematching SHARED ${MATCHER_SOURCEFILES} ${VERSIONINFO_RC})
        target_compile_definitions(templatematching PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching PRIVATE -DLIB_EXPORTS)

        add_library(templatematching_ctype SHARED ${MATCHER_C_TYPE_SOURCEFILES} ${VERSIONINFO_RC})
        target_compile_definitions(templatematching_ctype PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching_ctype PRIVATE -DLIB_EXPORTS)
    else()
        add_library(templatematching STATIC ${MATCHER_SOURCEFILES} ${VERSIONINFO_RC})
        add_library(templatematching_ctype STATIC ${MATCHER_C_TYPE_SOURCEFILES} ${VERSIONINFO_RC})
    endif()
else()
    if (BUILD_SHARED_LIBS)
        add_library(templatematching SHARED ${MATCHER_SOURCEFILES})
        target_compile_definitions(templatematching PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching PRIVATE -DLIB_EXPORTS)

        add_library(templatematching_ctype SHARED ${MATCHER_C_TYPE_SOURCEFILES} ${VERSIONINFO_RC})
        target_compile_definitions(templatematching_ctype PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching_ctype PRIVATE -DLIB_EXPORTS)
    else()
        add_library(templatematching STATIC ${MATCHER_SOURCEFILES})
        add_library(templatematching_ctype STATIC ${MATCHER_C_TYPE_SOURCEFILES} ${VERSIONINFO_RC})
    endif()
endif()


target_link_libraries(templatematching ${OpenCV_LIBS})
target_link_libraries(templatematching_ctype ${OpenCV_LIBS})


#安装规则
install(TARGETS templatematching 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS templatematching_ctype 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES 
    ${PROJECT_SOURCE_DIR}/include/matcher.h 
    ${PROJECT_SOURCE_DIR}/include/template_matching.h 
	TYPE INCLUDE 
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32)
install(FILES 
    ${PROJECT_SOURCE_DIR}/3rdParty/opencv/x64/vc16/bin/opencv_world490.dll
	TYPE BIN 
	DESTINATION ${CMAKE_INSTALL_BINDIR})
else()

endif()

