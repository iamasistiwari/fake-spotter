import React from 'react'
import Message from '../../../components/Message';

export default async function page() {
  return (
    <div className="h-screen w-full text-white bg-[#212121] flex flex-col ">
        <Message />
    </div>
  );
}
