#ifndef _BUTTERFLY_UTILS_H_
#define _BUTTERFLY_UTILS_H_

#ifndef INLINE
# if __STDC_VERSION__ >= 199901L
#  define INLINE inline
# else
#  define INLINE
# endif
#endif

static inline bfm_index_t read_index(char **head) {
  bfm_index_t result = *(bfm_index_t*)*head;
  *head += sizeof(bfm_index_t);
  return result;
}

static inline int64_t read_int64(char **head) {
  int64_t result = *(int64_t*)*head;
  *head += sizeof(int64_t);
  return result;
}

static inline void read_int64_list(char **head, int64_t *list, size_t n) {
  for (size_t i = 0; i != n; ++i) {
    list[i] = *(int64_t*)*head;
    *head += sizeof(int64_t);
  }
}

static inline void read_pointer_list(char **head, char **list, size_t n, char *baseptr) {
  for (size_t i = 0; i != n; ++i) {
    list[i] = baseptr + read_int64(head);
  }
}

#endif
