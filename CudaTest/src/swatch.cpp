#include "swatch.h"

StopWatch::StopWatch(void)
{
    LARGE_INTEGER LFreq;

    QueryPerformanceFrequency(&LFreq);
    FFreq = LFreq.HighPart * 4294967296.0 + LFreq.LowPart;
}

void StopWatch::Reset(void)
{
    FTime = 0;
}

void StopWatch::Start(void)
{
    QueryPerformanceCounter(&StartTime);
}

void StopWatch::Stop(void)
{
    double start, end;

    QueryPerformanceCounter(&EndTime);
    end   = EndTime.  HighPart * 4294967296.0 + EndTime.  LowPart;
    start = StartTime.HighPart * 4294967296.0 + StartTime.LowPart;
    FTime += (end - start) / FFreq;
}
