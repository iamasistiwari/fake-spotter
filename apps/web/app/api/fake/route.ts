/* eslint-disable @typescript-eslint/ban-ts-comment */
import { NextRequest, NextResponse } from "next/server"



export async function POST(req: NextRequest) {
    const data = await req.json()
    //@ts-ignore
    const videoId: string = data.videoId
    await new Promise((resolve) => setTimeout(resolve, 5000))
    return NextResponse.json({
        videoId: videoId,
        isFake: true,
        confidence: 95
    })
}