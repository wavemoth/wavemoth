#ifndef _FASTSHT_ERROR_H_
#define _FASTSHT_ERROR_H_

#define ERRMAXLEN 2048
static char error_buf[ERRMAXLEN];
#define _exception(file, line, msg) do { \
    fprintf(stderr, "%s:%d %s\n", file, line, msg); \
    abort(); \
  } while (0)
#define _exceptionf(file, line, msg, ...) do { \
    snprintf(error_buf, ERRMAXLEN, msg, __VA_ARGS__);                   \
    fprintf(stderr, "%s:%d %s\n", file, line, error_buf);                \
    abort(); \
  } while (0)

#define check(cond, msg) if (!(cond)) { _exception(__FILE__, __LINE__, msg); }
#define checkf(cond, msg, ...) if (!(cond)) { _exceptionf(__FILE__, __LINE__, msg, __VA_ARGS__); }

#endif
