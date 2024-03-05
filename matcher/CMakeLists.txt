list(APPEND DETECTOR_SOURCEFILES 
    boyanmatcher.cpp
    ../interface/boyanmatcher.h
    base_matcher/base_matcher.h
    base_matcher/base_matcher.cpp
    Pattern_Matching/PatternMatching.h
    Pattern_Matching/PatternMatching.cpp

)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
    set(VERSIONINFO_RC "${CMAKE_BINARY_DIR}/VersionInfo.rc")
    configure_file("${CMAKE_SOURCE_DIR}/VersionInfo.rc.in"
                   "${VERSIONINFO_RC}")
    if (BUILD_SHARED_LIBS)
        add_library(templatematching SHARED ${DETECTOR_SOURCEFILES} ${VERSIONINFO_RC})
        target_compile_definitions(templatematching PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching PRIVATE -DLIB_EXPORTS)
    else()
        add_library(templatematching STATIC ${DETECTOR_SOURCEFILES} ${VERSIONINFO_RC})
    endif()
else()
    if (BUILD_SHARED_LIBS)
        add_library(templatematching SHARED ${DETECTOR_SOURCEFILES})
        target_compile_definitions(templatematching PUBLIC -DLIB_SHARED_BUILD)
        target_compile_definitions(templatematching PRIVATE -DLIB_EXPORTS)
    else()
        add_library(templatematching STATIC ${DETECTOR_SOURCEFILES})
    endif()
endif()


target_link_libraries(templatematching ${OpenCV_LIBS})


#安装规则
install(TARGETS templatematching 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES 
    ${PROJECT_SOURCE_DIR}/interface/boyanmatcher.h 
    ${PROJECT_SOURCE_DIR}/interface/boyan_template_matching.h 
	TYPE INCLUDE 
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WIN32)
install(FILES 
    ${PROJECT_SOURCE_DIR}/3rdParty/opencv/x64/vc16/bin/opencv_world490.dll
	TYPE BIN 
	DESTINATION ${CMAKE_INSTALL_BINDIR})
else()

endif()

