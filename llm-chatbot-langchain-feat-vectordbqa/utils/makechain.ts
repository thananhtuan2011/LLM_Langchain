import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { LLMChain, loadQAChain, ChatVectorDBQAChain } from 'langchain/chains';
import { PromptTemplate } from 'langchain/prompts';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Bạn là trợ lý ảo . Bạn tìm câu trả lời trong dữ liệu và trả lời chính xác câu hỏi .
  Bạn là trợ lý ảo . Bạn tìm câu trả lời trong dữ liệu và trả lời chính xác câu hỏi .
  

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT =
  PromptTemplate.fromTemplate(`Bạn là một trợ lý AI hữu ích. Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi ở cuối.
  Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. KHÔNG cố gắng bịa ra một câu trả lời.
  Nếu câu hỏi nằm ngoài ngữ cảnh, hãy lịch sự trả lời rằng bạn chỉ trả lời những câu hỏi nằm trong ngữ cảnh đó. Yêu cầu trả lời ngữ cảnh bằng tiếng Việt.
​

{context}

Question: {question}
Helpful answer in markdown:`);

export const makeChain = (vectorstore: PineconeStore) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0, modelName: 'gpt-3.5-turbo' }),
    prompt: CONDENSE_PROMPT,
  });

  const docChain = loadQAChain(
    //change modelName to gpt-4 if you have access to it
    new OpenAI({ temperature: 0, modelName: 'gpt-3.5-turbo' }),
    {
      prompt: QA_PROMPT,
    },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 4, //number of source documents to return. Change this figure as required.
  });
};
