import { create } from 'zustand'

export type Message = {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface ChatState {
  messages: Message[]
  isStreaming: boolean
  addMessage: (msg: Message) => void
  setStreaming: (streaming: boolean) => void
  updateLastMessage: (content: string) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isStreaming: false,
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  updateLastMessage: (content) => set((state) => {
    const newMessages = [...state.messages]
    const lastIdx = newMessages.length - 1
    if (newMessages[lastIdx].role === 'assistant') {
      newMessages[lastIdx].content = content
    }
    return { messages: newMessages }
  })
}))
