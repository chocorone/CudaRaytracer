#pragma once
#include <windows.h>

class StopWatch
{
private:
        LARGE_INTEGER StartTime, EndTime;
        double FFreq;
        double FTime;
public:
        StopWatch(void);
        void Reset(void);
        void Start(void);
        void Stop(void);
        double GetFreq(void) { return FFreq; };
        double GetTime(void) { return FTime; };
};

