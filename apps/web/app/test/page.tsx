"use client"
import React from 'react'
import {IKUpload} from "imagekitio-next"


const authenticator = async () => {
  try {
    const res = await fetch(`/api/upload/video`)
    if(!res.ok){
      const errorText = await res.text()
      throw new Error(`Request failed with code ${res.status} and ${errorText}`)
    }
    const data = await res.json()
    const {signature, expire, token } = data
    return { signature, expire, token }; 
  } catch (error) {
    throw new Error(`Failed ${error}`)
  }
};

export default function page() {
  return (
    <div className="text-ctext h-screen">
      <IKUpload
        validateFile={(file) =>{
          console.log("THIS IS FILE", file)
          if(file.size > 1024*1024*100){
            // file should less than 100MB
            alert("File should less than 100mb")
            return false
          }
          const allowedTypes = ["video/mp4"];
          if(!allowedTypes.includes(file.type)){
            alert("Only mp4 are allowed")
            return false
          }
          return true
        }}
        onSuccess={(res) => {
          console.log("SUCCESS",res)
        }}
        onUploadProgress={(progress) => {
          console.log("PROGRESS", progress)
        }}
        onUploadStart={(det) => {
          console.log("ON START", det)
        }}
        urlEndpoint={`${process.env.NEXT_PUBLIC_URL_ENDPOINT}`}
        publicKey={`${process.env.NEXT_PUBLIC_KEY}`}
        authenticator={authenticator}
      />
    </div>
  );
}
