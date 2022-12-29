using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PhoneCamera : MonoBehaviour
{
    private bool CamAvailable;
    private WebCamTexture backCam;
    private Texture defaultBackground;

    public Button buttonShot;
    public RawImage background;
    public AspectRatioFitter fit;
    public GameObject ImageScreen;

    void Start()
    {
        defaultBackground = background.texture;
        WebCamDevice[] devices = WebCamTexture.devices;

        if(devices.Length == 0)
        {
            Debug.Log("No cam detected");
            CamAvailable = false;
            return;
        }

        for(int i = 0; i<devices.Length; i++)
        {
            if (!devices[i].isFrontFacing)
            {
                backCam = new WebCamTexture(devices[i].name, Screen.width, Screen.height);
            }
        }

        if(backCam == null)
        {
            Debug.Log("Unable to find back camera");
            return;
        }

        backCam.Play();
        background.texture = backCam;

        CamAvailable = true;
    }

    void Update()
    {
        if (!CamAvailable)
        {
            return;
        }

        float ratio = (float)backCam.width / (float)backCam.height;
        fit.aspectRatio = ratio;

        float scaleY = backCam.videoVerticallyMirrored ? -1f : 1f;
        background.rectTransform.localScale = new Vector3(1f, scaleY, 1f);

        int orient = -backCam.videoRotationAngle;
        background.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

        if (backCam.videoRotationAngle == 0 || backCam.videoRotationAngle == 180)
        {
            ImageScreen.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
        }
        else
        {
            ImageScreen.transform.localScale = new Vector3(1f, 1f, 1.0f);
        }
    }
}
