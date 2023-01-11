#include "pch.h"
#include "Test.h"

extern "C" int division(int x, int y) {
	if (y <= 0) {
		return -1;
	}
	return x / y;
}