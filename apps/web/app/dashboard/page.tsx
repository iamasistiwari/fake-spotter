/* eslint-disable @typescript-eslint/no-unused-vars */
import { getServerSession } from 'next-auth'
import React from 'react'
import { authOptions } from '../../lib/auth'
import Sidebar from '../../components/Sidebar'
import Message from '../../components/Message';

export default async function page() {
    // const session = await getServerSession(authOptions)
  return (
    <div className="h-screen bg-[#212121]">
      <Sidebar> 
        <Message />
      </Sidebar>
    </div>
  );
}
