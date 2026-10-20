import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function MarkdownText({text,className=''}:{text:string;className?:string}){
 return <div className={'markdown-text '+className}><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml components={{
   a:({children,href})=><a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
   img:({alt})=><span className="image-placeholder">{alt||'图片引用'}</span>,
   table:({children})=><div className="markdown-table"><table>{children}</table></div>
 }}>{text}</ReactMarkdown></div>;
}
