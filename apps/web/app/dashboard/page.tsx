import { getServerSession } from 'next-auth'
import React from 'react'
import { authOptions } from '../../lib/auth'

export default async function page() {
    const session = await getServerSession(authOptions)
  return (
    <div>{JSON.stringify(session)}</div>
  )
}
