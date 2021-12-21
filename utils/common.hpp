#include "windows.h"
#include <string>

namespace myutils
{
    class Common
    {
    public:
        static std::wstring GetModulePath()
        {
            wchar_t pBuf[256];
            wchar_t *p = &pBuf[0];
            size_t len = sizeof(pBuf);
            int bytes = GetModuleFileNameW(NULL, p, len);
            return std::wstring(pBuf, bytes);
        }
    };

}