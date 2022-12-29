using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class PhotoCapture : MonoBehaviour
{
    private Texture2D screenshot;
    public GameObject buttonScreen;
    public GameObject imageScreen;
    [SerializeField] private Image screenPhoto;
    [SerializeField] private string pathFile;
    [SerializeField] private string Filename;
    [SerializeField] private string time;

    public Text showError;

    // Start is called before the first frame update
    void Start()
    {
        imageScreen.SetActive(false);
        screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false); 
        
        pathFile = "/storage/emulated/0/Download/MachineLearningPicture";
        time = System.DateTime.UtcNow.ToLocalTime().ToString("dd-MM-yyyy_HH-mm-ss");
        if (Directory.Exists (pathFile) == false) {
            Directory.CreateDirectory(pathFile);
            showError.text = "things";
        } 
    }

    public void TakeAShot()
    {
        buttonScreen.SetActive(false);
        imageScreen.SetActive(false);
        Debug.Log("shot");
        StartCoroutine(CapturePhoto());
    }

    IEnumerator CapturePhoto()
    {
        yield return new WaitForEndOfFrame();

        Rect regionToRead = new Rect(0, 0, Screen.width, Screen.height);
        
        screenshot.ReadPixels(regionToRead, 0, 0, false);
        screenshot.Apply();
        ShowPhoto();
    }

    void ShowPhoto()
    {
        Sprite photoSprite = Sprite.Create(screenshot, new Rect(0.0f, 0.0f, screenshot.width, screenshot.height), new Vector2(0.5f, 0.5f),
            100.0f);
        screenPhoto.sprite = photoSprite;
        buttonScreen.SetActive(true);
        imageScreen.SetActive(true);
        CreateFile();
    }

    void CreateFile()
    {
        showError.text = "test";
        Filename = pathFile + "/Image_"+ time + ".png";
        byte[] pngBytes = screenshot.EncodeToPNG();
        File.WriteAllBytes(Filename,pngBytes);
        showError.text = "ok "+time;
    }
}
