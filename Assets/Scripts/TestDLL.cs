using DynamicLibraries;
using UnityEngine;

public class TestDLL : MonoBehaviour
{

    private void Start()
    {
        Debug.Log(DynamicLibWrapper.test());
    }
}
